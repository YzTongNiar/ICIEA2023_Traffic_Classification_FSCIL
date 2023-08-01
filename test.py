import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Class-Incremental Learning training and evaluation script', add_help=False)
    parser.add_argument('--num_bases', default=5, type=int)
    parser.add_argument('--increment', default=3, type=int)
    parser.add_argument('--num_tasks', default=16, type=int)
    parser.add_argument('--backbone', default="lstm", type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--input_size', default=8, type=int)
    parser.add_argument('--herding_method', default="barycenter", type=str)
    parser.add_argument('--memory_size', default=25, type=int)
    parser.add_argument('--fixed_memory', default=False, action="store_true")
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--data_set', default='cifar')
    parser.add_argument('--data_path', default='/data/data/data/cifar100')
    parser.add_argument('--lambda_kd', default=0.5, type=float)
    parser.add_argument('--dynamic_lambda_kd', action="store_true")
    parser.add_argument('--class_order', default=list(range(1, 21)))
    parser.add_argument('--known_classes', default=0)
    return parser

parser = get_args_parser()
args = parser.parse_args()
args.task_id = 0
print(args.task_id)