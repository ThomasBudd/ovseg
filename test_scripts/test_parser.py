import argparse

parser = argparse.ArgumentParser(description='Recieving inputs for training settings')

parser.add_argument('-w', '--weight_indx', type=int, help='an integer determining which loss is chosen '
                    'from the list', default=0)
parser.add_argument('-m', '--model_indx', type=int, help='an integer determining which model is chosen '
                    'from the list', default=0)

args = parser.parse_args()

loss_weight = [0.5, 0.7, 0.9, 1.0][args.weight_indx]
recon_model = ['recon_fbp_convs_full_HU', 'reconstruction_network_fbp_convs'][args.model_indx]

print('loss_weight: {}'.format(loss_weight))
print('recon_model: {}'.format(recon_model))

