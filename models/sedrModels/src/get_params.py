import argparse

def getSEDRParams():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
    parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                        help='graph distance type: euclidean/cosine/correlation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--cell_feat_dim', type=int, default=200, help='Dim of PCA')
    parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
    parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
    parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
    parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
    parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
    parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
    parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
    parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
    parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
    parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
    parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
    parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
    parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
    parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
    # ______________ Eval clustering Setting _________
    parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
    parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.') 

    params = parser.parse_args()
    return params

def getSEDRVAEParams():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train.')
    parser.add_argument('--cell_feat_dim', type=int, default=200, help='Dim of PCA')
    parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
    parser.add_argument('--feat_hidden2', type=int, default=100, help='Dim of DNN hidden 2-layer.')
    parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
    
    parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
    parser.add_argument('--kl_w', type=float, default=0.03, help='Weight of GCN loss.')
    parser.add_argument('--latent_dim', type=float, default=10, help='latent dim')
    
    parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial VAE learning rate.')
    parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
    parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
    parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
    parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
    # ______________ Eval clustering Setting _________
    parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
    parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.') 

    params = parser.parse_args()
    return params
