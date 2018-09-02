import sys
sys.path.insert(0, '/home/research/disk1/caffe/python')
import caffe
from caffe import layers as L,params as P,to_proto
import caffe.proto.caffe_pb2 as caffe_pb2
from google.protobuf.text_format import Merge
path='infos/'
train_list=path+'train.txt'
val_list=path+'val.txt'           
train_proto=path+'train_hg.prototxt'   
val_proto=path+'val_hg.prototxt'
train_proto_mobile=path+'train_hgm.prototxt'
val_proto_mobile=path+'val_hgm.prototxt'

def conv_bn_relu(input, kernel_size, num_output, stride, pad):
    conv = L.Convolution(input, kernel_size=kernel_size, stride=stride, num_output=num_output, pad=pad,
                         weight_filler=dict(type='xavier'))
    bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

def conv_block_moible(input, num_output, multi):
    conv1 = L.Convolution(input, kernel_size=1, num_output=num_output*multi, stride=1, pad=0,
                          weight_filler=dict(type='xavier'))
    bn1 = L.BatchNorm(conv1, use_global_stats=False, in_place=True)
    scale1 = L.Scale(bn1, bias_term=True, in_place=True)
    relu1 = L.ReLU(scale1, in_place=True)
    conv2 = L.Convolution(relu1, kernel_size=3, num_output=num_output*multi, stride=1, pad=1,
                          group=num_output*multi, weight_filler=dict(type='xavier'))
    bn2 = L.BatchNorm(conv2, use_global_stats=False, in_place=True)
    scale2 = L.Scale(bn2, bias_term=True, in_place=True)
    relu2 = L.ReLU(scale2, in_place=True)
    conv3 = L.Convolution(relu2, kernel_size=1, num_output=num_output, stride=1, pad=0,
                          weight_filler=dict(type='xavier'))
    bn3 = L.BatchNorm(conv3, use_global_stats=False, in_place=True)
    scale3 = L.Scale(bn3, bias_term=True, in_place=True)
    return scale3

def skip_layer(input, num_output, num_input):
    if num_output == num_input:
        return input
    else:
        conv = L.Convolution(input, kernel_size=1, num_output=num_output, stride=1, pad=0,
                             weight_filler=dict(type='xavier'))
        return conv

def residual_mobile(input, num_output, multi, num_input):
    convb = conv_block_moible(input, num_output, multi)
    skipl = skip_layer(input, num_output, num_input)
    addn  = L.Eltwise(convb, skipl, eltwise_param=dict(operation=1))
    return addn

def up_sample(input, num_output):
    return L.Deconvolution(input, param=[dict(lr_mult=0, decay_mult=0)], 
        convolution_param=dict(bias_term=False, num_output=num_output, kernel_size=2, stride=2,
            group=num_output, pad=0, weight_filler=dict(type='bilinear')))

def deconv(input, num_output, kernel_size, stride, pad):
    return L.Deconvolution(input, convolution_param=dict(bias_term=False, num_output=num_output,
                            kernel_size=kernel_size, stride=stride, pad=pad))

def hourglass_mobile(input, num_output, num_modual, multi, num_input):
    up_1 = residual_mobile(input, num_output, multi, num_input)
    low = L.Pooling(input, pool=P.Pooling.MAX, stride=2, kernel_size=2)
    low_1 = residual_mobile(low, num_output, multi, num_input=num_output)
    if num_modual > 1:
        low_2 = hourglass_mobile(low_1, num_output, num_modual-1, multi, num_input=num_output)
    else:
        low_2 = residual_mobile(low_1, num_output, multi, num_input=num_output)
    low_3 = residual_mobile(low_2, num_output, multi, num_input=num_output)
    up_2 = up_sample(low_3, num_output)
    return L.Eltwise(up_2, up_1, eltwise_param=dict(operation=1))

def stacked_hourglass_network(batch_size, img_size, nfeats, multi, out_dim, include_acc=False):
    data, label = L.MemoryData(batch_size=batch_size, channels=3, height=img_size, width=img_size, 
                                ntop=2, include=dict(phase=0))
    data = L.Input()
    conv1 = conv_bn_relu(data, kernel_size=3, num_output=32, stride=2, pad=1)
    r1 = residual_mobile(conv1, num_output=32, multi=2, num_input=32)
    pool1 = L.Pooling(r1, pool=P.Pooling.MAX, stride=2, kernel_size=2)
    r3 = residual_mobile(pool1, num_output=nfeats, multi=multi, num_input = 32)
    #
    hg = hourglass_mobile(r3, num_output=nfeats, num_modual=4, multi=multi, num_input=nfeats)
    hgr = residual_mobile(hg, num_output=nfeats, multi=multi, num_input=nfeats)
    ll = conv_bn_relu(hgr, kernel_size=1, num_output=nfeats, stride=1, pad=0)
    out = deconv(ll, num_output=out_dim, kernel_size=4, stride=2, pad=1)

    loss = L.SigmoidCrossEntropyLoss(out, label)
    if include_acc:
        acc = L.Accuracy(out, label, include=dict(phase=1))
        return to_proto(loss, acc)
    else:
        return to_proto(loss)
    # return to_proto(loss)

def transfer2depthwise(proto_src, proto_dst):
    net = caffe_pb2.NetParameter()
    Merge(open(proto_src, 'r').read(), net)
    for layer in net.layer:
        if layer.type == "Convolution":
            if layer.convolution_param.group !=1:
                layer.type = "DepthwiseConvolution"
    with open(proto_dst, 'w') as tf:
        tf.write(str(net))

def write_hg_net():
    batch, size, nf, multi, nj = 10, 128, 32, 6, 74
    with open(train_proto, 'w') as f:
        f.write(str(stacked_hourglass_network(batch_size=batch, img_size=size, nfeats=nf, multi=multi, out_dim=nj)))
    transfer2depthwise(train_proto, train_proto_mobile)

    with open(val_proto, 'w') as f:
        f.write(str(stacked_hourglass_network(batch_size=batch, img_size=size, nfeats=nf, multi=multi, out_dim=nj, include_acc=True)))
    transfer2depthwise(val_proto, val_proto_mobile)

def create_net(img_list,batch_size,include_acc=True):
    # data,label=L.ImageData(source=img_list,batch_size=batch_size,new_width=48,new_height=48,ntop=2,
    #                        transform_param=dict(crop_size=40,mirror=True))
    # data,label=L.MemoryData(batch_size=batch_size,channels=3,height=256,width=256,ntop=2,
    #                         include=dict(phase=0))

    conv1=L.Convolution(data, kernel_size=5, stride=1,num_output=16, pad=2,weight_filler=dict(type='xavier'))
    relu1=L.ReLU(conv1, in_place=True)
    pool1=L.Pooling(relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2=L.Convolution(pool1, kernel_size=53, stride=1,num_output=32, pad=1,weight_filler=dict(type='xavier'))
    relu2=L.ReLU(conv2, in_place=True)
    pool2=L.Pooling(relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv3=L.Convolution(pool2, kernel_size=53, stride=1,num_output=32, pad=1,weight_filler=dict(type='xavier'))
    relu3=L.ReLU(conv3, in_place=True)
    pool3=L.Pooling(relu3, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    fc4=L.InnerProduct(pool3, num_output=1024,weight_filler=dict(type='xavier'))
    relu4=L.ReLU(fc4, in_place=True)
    drop4 = L.Dropout(relu4, in_place=True)
    fc5 = L.InnerProduct(drop4, num_output=7,weight_filler=dict(type='xavier'))
    # loss = L.SoftmaxWithLoss(fc5, label)
    loss = L.SigmoidCrossEntropyLoss(fc5, label)
    
    if include_acc:             
        acc = L.Accuracy(fc5, label, top_k=5, name='loss', include=dict(phase=1))
        return to_proto(loss, acc)
    else:
        return to_proto(loss)
    
def write_net():
    #
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_list,batch_size=64)))
    #    
    with open(val_proto, 'w') as f:
        f.write(str(create_net(val_list,batch_size=32, include_acc=True)))
        
if __name__ == '__main__':
    # write_net()
    write_hg_net()
