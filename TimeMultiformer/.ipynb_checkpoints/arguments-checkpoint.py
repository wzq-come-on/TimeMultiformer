import argparse
import os

def TimeMultiformer_arguments():
    '''
    Network Parameter Definition
    Modification and definition of global network parameters.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=['train', 'test', 'impute'], default='train', help="Model stage")
    parser.add_argument("--saving_impute_path", type=str, default='data/physio2012_37feats_01masked',
                        help="Path to save the imputed results")
    parser.add_argument('--data_path', type=str, choices=[
        'data/AirQuality_seqlen24_01masked',
        'data/Electricity_seqlen100_01masked',
        'data/physio2012_37feats_01masked',
        'data/ETTm1_seqlen24_01masked',
    ], default='data/ETTm1_seqlen24_01masked', help="Select the dataset")
    parser.add_argument("--seq_len", type=int, default=48, help="time series length in the dataset")
    parser.add_argument("--feature_num", type=int, default=37, help="The number of features in the dataset")
    parser.add_argument('--seed', type=int, default=2023, help='Construct the missing rate of the missing data')
    parser.add_argument('--model', type=str, choices=['TimeMultiformer'],\
                        default='TimeMultiformer', help="Model selection")

    parser.add_argument("--num_workers", type=int, default=1, help="The number of subprocesses for data loading in dataloader")
    parser.add_argument("--MIT", type=bool, default=True, help="Whether to perform masked_imputation_task")
    parser.add_argument("--model_type", type=str, default='TimeMultiformer', help="Model type affects the data reading method")
    parser.add_argument("--device", type=str, default='cuda', help="Whether to use cuda")

    parser.add_argument("--epochs", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--epochs_G", type=int, default=1, help="生成器迭代次数")
    parser.add_argument("--batch_size", type=int, default=128, help="Small batch size")
    parser.add_argument("--lr", type=float, default=0.000682774550436755, help="Learning rate ")

    parser.add_argument("--optimizer_type", type=str, default='adam', help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Optimizer parameters")
    
    parser.add_argument("--n_groups", type=int, default=2, help="The number of Ne groups in Feature Dependencies Learning")
    parser.add_argument("--n_group_inner_layers", type=int, default=1, help="The number of layers of Temporal Dependencies Learning Net")
    parser.add_argument("--d_model", type=int, default=256, help="Multi-head attention mechanism embedding mapping features")
    parser.add_argument("--d_inner", type=int, default=512, help="PositionWiseFeedForward parameter")
    parser.add_argument("--n_head", type=int, default=16, help="The number of parameter heads in the multi-head attention mechanism")
    parser.add_argument("--d_k", type=int, default=32, help="The parameter q of the multi-head attention mechanism, the dimension of the k layer")
    parser.add_argument("--d_v", type=int, default=32, help="The dimension of the parameter v layer of the multi-head attention mechanism")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout probability")
    parser.add_argument("--diagonal_attention_mask", type=bool, default=True, help="Whether it is a diagonal mask")

    parser.add_argument('--miss_rate', type=float, default=0.1, help='Construct the missing rate of the missing data')
    parser.add_argument('--alpha', type=list, default=[100,100], help='Loss function combined with proportion')
    parser.add_argument('--lambda_gp', type=int, default=10, help='Proportion of penalty items')

    parser.add_argument('--saving_model_path', type=str, default='./SavedModel', help='Model saving directory')
    parser.add_argument('--best_imputation_MAE', type=float, default=1.0, help='Optimal strategy value of the model')
    parser.add_argument('--best_imputation_MAE_Threshold', type=float, default=0.5, help='The model saves the optimal policy threshold')
    parser.add_argument('--best_imputation_RMSE', type=float, default=2.0, help='Optimal strategy value of the model')
    parser.add_argument('--best_imputation_RMSE_Threshold', type=float, default=2.0, help='The model saves the optimal policy threshold')
    parser.add_argument('--best_imputation_MRE', type=float, default=2.0, help='Optimal strategy value of the model')
    parser.add_argument('--best_imputation_MRE_Threshold', type=float, default=2.0, help='The model saves the optimal policy threshold')
    parser.add_argument('--min_mae_loss', type=float, default=0.5, help='Model verification threshold')

    parser.add_argument('--log_saving', type=str, default='./logs', help='Log storage directory')

    args = parser.parse_args()
    if args.data_path=='data/physio2012_37feats_01masked':
        args.seq_len=48
        args.feature_num=37
        args.saving_model_path = os.path.join(args.saving_model_path, args.model)
        args.saving_model_path = os.path.join(args.saving_model_path, "physio2012")
        args.log_saving = os.path.join(args.log_saving, args.model)
        args.log_saving = os.path.join(args.log_saving, "physio2012")
        args.d_inner = 512
        args.d_k = 32
        args.d_model = 512
        args.d_v = 64
        args.n_group_inner_layers = 1
        args.n_groups = 5
        args.n_head = 8
        args.lr = 0.0005
        args.dropout = 0
        args.epochs = 1000

    if args.data_path=='data/AirQuality_seqlen24_01masked':
        args.seq_len=24
        args.feature_num=132
        args.saving_model_path = os.path.join(args.saving_model_path, args.model)
        args.saving_model_path = os.path.join(args.saving_model_path, "AirQuality")
        args.log_saving = os.path.join(args.log_saving, args.model)
        args.log_saving = os.path.join(args.log_saving, "AirQuality")
        args.d_inner = 512
        args.d_k = 128
        args.d_model = 256
        args.d_v = 64
        args.n_group_inner_layers = 1
        args.n_head = 4
        args.lr = 0.0001
        args.dropout = 0.1
        args.miss_rate = 0.1
        args.epochs = 10000

    if args.data_path=='data/Electricity_seqlen100_01masked':
        args.seq_len=100
        args.feature_num=370
        args.saving_model_path = os.path.join(args.saving_model_path, args.model)
        args.saving_model_path = os.path.join(args.saving_model_path, "Electricity")
        args.log_saving = os.path.join(args.log_saving, args.model)
        args.log_saving = os.path.join(args.log_saving, "Electricity")
        args.d_inner = 128
        args.d_k = 128
        args.d_model = 2048
        args.d_v = 128
        args.n_group_inner_layers = 3
        args.n_head = 8
        args.dropout=0.0
        args.lr = 0.0002
        args.miss_rate = 0.1
        args.epochs = 2000

    if args.data_path=='data/ETTm1_seqlen24_01masked':
        args.seq_len=24
        args.feature_num=7
        args.saving_model_path = os.path.join(args.saving_model_path, args.model)
        args.saving_model_path = os.path.join(args.saving_model_path, "ETT")
        args.log_saving = os.path.join(args.log_saving, args.model)
        args.log_saving = os.path.join(args.log_saving, "ETT")
        args.n_group_inner_layers = 5
        args.lr = 0.001
        args.hint_rate = 0.1
        args.miss_rate = 0.1
        args.d_model = 256
        args.d_inner = 512
        args.n_head = 1
        args.d_k = 32
        args.d_v = 32
        args.epochs = 10000

    return args

if __name__ == '__main__':
    args = TimeMultiformer_arguments()

    import pandas as pd

    alllist = []
    column = ['one', 'two']


    print('--------args----------')
    for k in list(vars(args).keys()):
        list1=[k, vars(args)[k]]
        alllist.append(list1)
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')
    test = pd.DataFrame(columns=column, data=alllist)
    test.to_csv('test.csv')
