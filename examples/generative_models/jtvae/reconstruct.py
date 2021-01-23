import rdkit
import torch

from dgllife.data import JTVAEZINC, JTVAEDataset, JTVAECollator
from dgllife.utils import JTVAEVocab
from dgllife.model import JTNNVAE
from torch.utils.data import DataLoader

def main(args):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    if args.use_cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    vocab = JTVAEVocab(file_path=args.train_path)
    if args.test_path is None:
        dataset = JTVAEZINC('test', vocab)
    else:
        dataset = JTVAEDataset(args.test_path, vocab, training=False)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            collate_fn=JTVAECollator(training=False))

    model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depth, None)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model = model.to(device)

    acc = 0.0
    for it, (tree, tree_graph, mol_graph) in enumerate(dataloader):
        tot = it + 1
        smiles = tree.smiles
        tree_graph = tree_graph.to(device)
        mol_graph = mol_graph.to(device)
        dec_smiles = model.reconstruct(tree_graph, mol_graph)
        if dec_smiles == smiles:
            acc += 1
        if tot % args.print_iter == 0:
            print('Iter {:d}/{:d} | Acc {:.4f}'.format(
                tot // args.print_iter, len(dataloader) // args.print_iter, acc / tot))
    print('Final acc: {:.4f}'.format(acc / tot))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-tr', '--train-path', type=str,
                        help='Path to the training molecules, with one SMILES string a line')
    parser.add_argument('-te', '--test-path', type=str,
                        help='Path to the test molecules, with one SMILES string a line')
    parser.add_argument('-m', '--model-path', type=str,
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('-w', '--hidden-size', type=int, default=450,
                        help='Hidden size')
    parser.add_argument('-l', '--latent-size', type=int, default=56,
                        help='Latent size')
    parser.add_argument('-d', '--depth', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('-pi', '--print-iter', type=int, default=20,
                        help='Frequency for printing evaluation metrics')
    parser.add_argument('-cpu', '--use-cpu', action='store_true',
                        help='By default, the script uses GPU whenever available. '
                             'This flag enforces the use of CPU.')
    args = parser.parse_args()

    main(args)
