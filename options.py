import argparse

def arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--clamp",default=2.0,type=float,help='a constant factor',required=False
    )
    parser.add_argument(
        "--device_ids", default=[0], type=int, help='a constant factor', required=False
    )
    parser.add_argument(
        "--init_scale",default=0.01,type=float,help=" init_weight_scale for model initial",required=False
    )
    parser.add_argument(
        '--lr',default=10 ** -4.5,type=float,help='setting learning rate',required=False
    )
    parser.add_argument(
        '--betas',default=(0.5, 0.999),type=tuple,help='betas setting for Adam',required=False
    )
    parser.add_argument(
        '--eps',default=1e-6,type=float,help='prevent zero',required=False
    )
    parser.add_argument(
        '--weight_decay',default=1e-5,type=float,help='weight decay',required=False
    )
    parser.add_argument(
        '--weight_step',default=1000,type=int,help='Learning rate decay interval',required=False
    )
    parser.add_argument(
        '--gamma',default=0.5,type=float,help='current lr = original lr*gamma',required=False
    )
    parser.add_argument(
        '--channels_in',default=3,type=int,help='the input channels',required=False
    )
    parser.add_argument(
        '--lamda_reconstruction',default=3,type=int,help='the weight for balancing secret-rev',required=False
    )
    parser.add_argument(
        '--lamda_guide',default=1,type=int,help='the weight for balancing stego',required=False
    )
    parser.add_argument(
        '--lamda_low_F',default=1,type=int,help='the weight for balaning stego in low frequency',required=False
    )
    parser.add_argument(
        '--train_image_dir',default=' ',type=str,help='the image path for training',required=False
    )
    parser.add_argument(
        '--image_patch_size',default=512,type=int,help='training patch size',required=False
    )
    parser.add_argument(
        '--training_batch',default=2,type=int,help='the training batchsize',required=False
    )
    parser.add_argument(
        '--epoch',default=20000,type=int,help='the total training epoch',required=False
    )
    parser.add_argument(
        '--save_frequency',default=1000,type=int,help=' save model per ? epochs',required=False
    )
    parser.add_argument(
        '--model_path',default='C:/HyyData/model/model20.pt',help='save your model to this path',required=False
    )
    parser.add_argument(
        '--val_frequency',default=100,type=int,help='validate your model per ? epochs',required=False
    )
    parser.add_argument(
        '--testpath',default='C:/datasets/face_valid/',type=str,help='test image path',required=False
    )
    parser.add_argument(
        '--cropsize_test',default=512,type=int,help=' the size of the test image',required=False
    )
    parser.add_argument(
        '--test_mask_face_path',default='C:/HyyData/image/mask_face/'
    )
    parser.add_argument(
        '--test_test_protect_face_path',default='C:/HyyData/image/protect_face/'
    )
    parser.add_argument(
        '--test_masked_face_path',default='C:/HyyData/image/masked_face/'
    )
    parser.add_argument(
        '--test_protect_facet_rev_path',default='C:/HyyData/image/protect-rev/'
    )
    parser.add_argument(
        '--test_mask_face_rev_path',default='C:/HyyData/image/mask-rev/'
    )
    parser.add_argument(
        '--test_resi_mask_and_masked_path', default='C:/HyyData/image/resi_mask/'
    )
    parser.add_argument(
        '--test_resi_protected_and_recovered_path', default='C:/HyyData/image/resi_protect/'
    )
    parser.add_argument(
        '--test_ramdom_matrix_path', default='C:/HyyData/image/random_matrix/'
    )
    parser.add_argument(
        '--test_lost_matrix_m_path', default='C:/HyyData/image/lost_matrix/'
    )

    opt=parser.parse_args()
    return opt