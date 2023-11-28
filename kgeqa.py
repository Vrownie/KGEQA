import torch; print("imported torch")
import torch_geometric; print("imported torch_geometric")

import random

try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup

from transformers import AutoTokenizer

from modeling.modeling_kgeqa import *
from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *


DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'csqa_extract': 1e-3,
    'squad': 1e-3,
    'squad1': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}

HAS_UNANSWERABLE = ['squad']

NO_TEST_SET = ['squad', 'squad1']

from collections import defaultdict, OrderedDict
import numpy as np

import socket, os, subprocess, datetime
print(socket.gethostname())
print ("pid:", os.getpid())
print ("conda env:", os.environ['CONDA_DEFAULT_ENV'])
print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))
print ("gpu: %s" % subprocess.check_output('echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))


def evaluate_accuracy(eval_set, model, give_example, has_unanswerable):
    n_samples, total_em, total_f1 = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            logits, _ = model(*input_data)
            s_e = logits.squeeze(1).argmax(1)
            len_pred = torch.relu(s_e[..., 1] - s_e[..., 0])
            len_true = torch.relu(labels[..., 1] - labels[..., 0])
            len_match = torch.relu(torch.min(s_e[..., 1], labels[..., 1]) - torch.max(s_e[..., 0], labels[..., 0]))
            precision, recall = len_match / len_pred, len_match / len_true
            f1 = (2 * precision * recall / (precision + recall)).nan_to_num(0)
            em = torch.all(s_e == labels, -1).float()
            if has_unanswerable: 
                unanswerable_bools = (s_e[..., 1] < s_e[..., 0]).float()
                cond = torch.all(labels < 0, -1)
                f1 = torch.where(cond, unanswerable_bools, f1)
                em = torch.where(cond, unanswerable_bools, em)
            total_f1 += f1.sum().item()
            total_em += em.sum().item()
            n_samples += labels.size(0)
    return total_em / n_samples, total_f1 / n_samples


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/kgeqa/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model_path', default=None)


    # data
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')

    # model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')

    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')


    # regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--fp16', default=False, type=bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()
    args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval_detail':
        # raise NotImplementedError
        eval_detail(args)
    else:
        raise ValueError('Invalid mode')




def train(args):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    loss_log_path = os.path.join(args.save_dir, 'loss_log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,dev_em,dev_f1,test_em,test_f1\n')
    with open(loss_log_path, 'w') as fout:
        fout.write('step,training_loss,ms_per_batch\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    # try:
    if True:
        if torch.cuda.device_count() > 0 and args.cuda > 0:
            device0 = torch.device(f"cuda:{args.cuda}")
            device1 = torch.device(f"cuda:{args.cuda}")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
        dataset = LM_KGEQA_DataLoader(args, args.train_statements, args.train_adj,
                                               args.dev_statements, args.dev_adj,
                                               args.test_statements, args.test_adj,
                                               batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                               device=(device0, device1),
                                               model_name=args.encoder,
                                               max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                               is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                               subsample=args.subsample, use_cache=args.use_cache)

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################
        no_ans = args.dataset in HAS_UNANSWERABLE
        if no_ans: 
            print("dataset has unanswerable questions! these will have special label [-1,-1]")
        print ('args.num_relation', args.num_relation)
        model = LM_KGEQA(args, args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation, n_concept=concept_num,
                                   concept_dim=args.gnn_dim, concept_in_dim=concept_dim,
                                   n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                                   p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf, seq_len=args.max_seq_len,
                                   pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                                   init_range=args.init_range, encoder_config={}, has_unanswerable=False) # TODO: ?
        if args.load_model_path:
            print (f'loading and initializing model from {args.load_model_path}')
            model_state_dict, old_args = torch.load(args.load_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(model_state_dict)

        model.encoder.to(device0)
        model.decoder.to(device1)


    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    def compute_loss(logits, labels, has_unanswerable):
        loss_func = nn.CrossEntropyLoss(reduction='mean')
        assert logits.shape[0] == 1 and labels.shape[0] == 1 # TODO: fine for now
        if has_unanswerable and labels[0][0] < 0 and labels[0][1] < 0: 
            new_labels = torch.argmax(logits.squeeze(1), -2)
            if new_labels[0][1] <= new_labels[0][0]:
                new_labels = torch.flip(new_labels,(-1,))
        else:
            new_labels = labels
        assert new_labels[0][0] >= 0 and new_labels[0][1] >= 0
        s_loss = loss_func(logits.squeeze(1)[..., 0], new_labels[..., 0])
        e_loss = loss_func(logits.squeeze(1)[..., 1], new_labels[..., 1])
        return (s_loss + e_loss) / 2

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
        
    print()
    print('-' * 71)
    if args.fp16:
        print ('Using fp16 training')
        scaler = torch.cuda.amp.GradScaler()

    global_step, best_dev_epoch = 0, 0
    best_dev_f1, final_test_f1, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    freeze_net(model.encoder)
    for epoch_id in range(args.n_epochs):
        if epoch_id == args.unfreeze_epoch:
            unfreeze_net(model.encoder)
        if epoch_id == args.refreeze_epoch:
            freeze_net(model.encoder)
        model.train()
        for qids, labels, *input_data in dataset.train():
            optimizer.zero_grad()
            bs = labels.size(0)
            for a in range(0, bs, args.mini_batch_size):
                b = min(a + args.mini_batch_size, bs)
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                        loss = compute_loss(logits, labels[a:b], no_ans)
                else:
                    logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                    loss = compute_loss(logits, labels[a:b], no_ans)
                loss = loss * (b - a) / bs
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += loss.item()
            if args.max_grad_norm > 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} |  loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, total_loss, ms_per_batch))
                with open(loss_log_path, 'a') as fout:
                    fout.write('{},{},{}\n'.format(global_step, total_loss, ms_per_batch))
                total_loss = 0
                start_time = time.time()
            global_step += 1

        model.eval()
        dev_em, dev_f1 = evaluate_accuracy(dataset.dev(), model, False, no_ans)
        save_test_preds = (epoch_id % 5 == 0)
        if not save_test_preds:
            test_em, test_f1 = evaluate_accuracy(dataset.test(), model, True, no_ans) if args.test_statements else (0.0, 0.0)
        else:
            eval_set = dataset.test()
            total_em, total_f1 = [], []
            count = 0
            preds_path = os.path.join(args.save_dir, 'kgeqa_test_e{}_preds.csv'.format(epoch_id))
            print(f"saving test set predictions to {preds_path}")
            with open(preds_path, 'w') as f_preds:
                print ('"{}","{}","{}","{}"'.format("qid", "question", "true_answer", "prediction"), file=f_preds)
                with torch.no_grad():
                    for qids, labels, *input_data in tqdm(eval_set):
                        logits, _, sent_vecs = model(*input_data, detail=True)
                        s_e = logits.squeeze(1).argmax(1)
                        for qid, label, start_end, sent_vec in zip(qids, labels, s_e, sent_vecs):
                            count += 1
                            len_pred = max(start_end[1] - start_end[0], 0.00001)
                            len_true = max(label[1] - label[0], 0.00001)
                            len_match = max(min(start_end[1], label[1]) - max(start_end[0], label[0]), 0.00001)
                            precision, recall = len_match / len_pred, len_match / len_true
                            if no_ans and label[0] < 0 and label[1] < 0:
                                unanswerable_bool = (start_end[1] < start_end[0]).float()
                                total_f1.append(unanswerable_bool)
                                total_em.append(unanswerable_bool)
                            else:
                                total_f1.append(2 * precision * recall / (precision + recall) if precision + recall != 0 else 0)
                                total_em.append(float(start_end[0] == label[0] and start_end[1] == label[1]))
                            dat_list = sent_vec.squeeze().tolist()
                            q_str = tokenizer.decode(dat_list, skip_special_tokens=True)
                            if no_ans and label[1] < 0 and label[0] < 0:
                                a_str = 'no answer'
                            else:
                                ss = min(len(dat_list), max(0, label[0]))
                                ee = min(len(dat_list), max(0, label[1]+1))
                                a_str = tokenizer.decode(dat_list[ss:ee])
                            if no_ans and start_end[1] < start_end[0]:
                                p_str = 'no answer'
                            else:
                                sss = min(len(dat_list), max(0, start_end[0]))
                                eee = min(len(dat_list), max(0, start_end[1]+1))
                                p_str = tokenizer.decode(dat_list[sss:eee])
                            print ('"{}","{}","{}","{}"'.format(qid, q_str, a_str, p_str), file=f_preds)
            f_preds.close()
            test_em, test_f1 = sum(total_em) / count, sum(total_f1) / count

        print('-' * 71)
        print('| epoch {:3} | step {:5} | dev_em {:7.4f} | dev_f1 {:7.4f} | test_em {:7.4f} | test_f1 {:7.4f} |'.format(epoch_id, global_step, dev_em, dev_f1, test_em, test_f1))
        print('-' * 71)
        with open(log_path, 'a') as fout:
            fout.write('{},{},{},{},{}\n'.format(global_step, dev_em, dev_f1, test_em, test_f1))
        if dev_f1 >= best_dev_f1:
            best_dev_f1 = dev_f1
            final_test_f1 = test_f1
            best_dev_epoch = epoch_id
            if args.save_model:
                torch.save([model.state_dict(), args], f"{model_path}.{epoch_id}")
                # with open(model_path +".{}.log.txt".format(epoch_id), 'w') as f:
                #     for p in model.named_parameters():
                #         print (p, file=f)
                print(f'model saved to {model_path}.{epoch_id}')
        else:
            if args.save_model:
                torch.save([model.state_dict(), args], f"{model_path}.{epoch_id}")
                # with open(model_path +".{}.log.txt".format(epoch_id), 'w') as f:
                #     for p in model.named_parameters():
                #         print (p, file=f)
                print(f'model saved to {model_path}.{epoch_id}')
        model.train()
        start_time = time.time()
        # if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            # break


def eval_detail(args):
    assert args.load_model_path is not None
    model_path = args.load_model_path

    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    model_state_dict, old_args = torch.load(model_path, map_location=torch.device('cpu'))
    model = LM_KGEQA(old_args, old_args.encoder, k=old_args.k, n_ntype=4, n_etype=old_args.num_relation, n_concept=concept_num,
                               concept_dim=old_args.gnn_dim, concept_in_dim=concept_dim,
                               n_attention_head=old_args.att_head_num, fc_dim=old_args.fc_dim, n_fc_layer=old_args.fc_layer_num,
                               p_emb=old_args.dropouti, p_gnn=old_args.dropoutg, p_fc=old_args.dropoutf, seq_len=old_args.max_seq_len,
                               pretrained_concept_emb=cp_emb, freeze_ent_emb=old_args.freeze_ent_emb,
                               init_range=old_args.init_range,
                               encoder_config={})
    model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() > 0 and args.cuda > 0:
        device0 = torch.device(f"cuda:{args.cuda}")
        device1 = torch.device(f"cuda:{args.cuda}")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    model.encoder.to(device0)
    model.decoder.to(device1)
    model.eval()

    statement_dic = {}
    for statement_path in (args.train_statements, args.dev_statements, args.test_statements):
        statement_dic.update(load_statement_dict(statement_path))

    use_contextualized = 'lm' in old_args.ent_emb

    print ('inhouse?', args.inhouse)

    print ('args.train_statements', args.train_statements)
    print ('args.dev_statements', args.dev_statements)
    print ('args.test_statements', args.test_statements)
    print ('args.train_adj', args.train_adj)
    print ('args.dev_adj', args.dev_adj)
    print ('args.test_adj', args.test_adj)

    dataset = LM_KGEQA_DataLoader(args, args.train_statements, args.train_adj,
                                           args.dev_statements, args.dev_adj,
                                           args.test_statements, args.test_adj,
                                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                           device=(device0, device1),
                                           model_name=old_args.encoder,
                                           max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
                                           is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                           subsample=args.subsample, use_cache=args.use_cache)

    tokenizer = AutoTokenizer.from_pretrained(old_args.encoder, use_fast=True)
    no_ans = HAS_UNANSWERABLE[args.dataset]
    if no_ans: 
        print("dataset has unanswerable questions! these will have special label [-1,-1]")

    save_test_preds = args.save_model
    dev_em, dev_f1 = evaluate_accuracy(dataset.dev(), model, False, no_ans)
    print('dev_em {:7.4f} | dev_f1 {:7.4f}'.format(dev_em, dev_f1))
    if not save_test_preds:
        test_em, test_f1 = evaluate_accuracy(dataset.test(), model, True, no_ans) if args.test_statements else (0.0, 0.0)
    else:
        eval_set = dataset.test()
        total_em, total_f1 = [], []
        count = 0
        preds_path = os.path.join(args.save_dir, 'kgeqa_test_e{}_preds.csv'.format(epoch_id))
        print(f"saving test set predictions to {preds_path}")
        with open(preds_path, 'w') as f_preds:
            print ('"{}","{}","{}","{}"'.format("qid", "question", "true_answer", "prediction"), file=f_preds)
            with torch.no_grad():
                for qids, labels, *input_data in tqdm(eval_set):
                    logits, _, sent_vecs = model(*input_data, detail=True)
                    s_e = logits.squeeze(1).argmax(1)
                    for qid, label, start_end, sent_vec in zip(qids, labels, s_e, sent_vecs):
                        count += 1
                        len_pred = max(start_end[1] - start_end[0], 0.00001)
                        len_true = max(label[1] - label[0], 0.00001)
                        len_match = max(min(start_end[1], label[1]) - max(start_end[0], label[0]), 0.00001)
                        precision, recall = len_match / len_pred, len_match / len_true
                        if no_ans and label[0] < 0 and label[1] < 0:
                            unanswerable_bool = (start_end[1] < start_end[0]).float()
                            total_f1.append(unanswerable_bool)
                            total_em.append(unanswerable_bool)
                        else:
                            total_f1.append(2 * precision * recall / (precision + recall) if precision + recall != 0 else 0)
                            total_em.append(float(start_end[0] == label[0] and start_end[1] == label[1]))
                        dat_list = sent_vec.squeeze().tolist()
                        q_str = tokenizer.decode(dat_list, skip_special_tokens=True)
                        if no_ans and label[1] < 0 and label[0] < 0:
                            a_str = 'no answer'
                        else:
                            ss = min(len(dat_list), max(0, label[0]))
                            ee = min(len(dat_list), max(0, label[1]+1))
                            a_str = tokenizer.decode(dat_list[ss:ee])
                        if no_ans and start_end[1] < start_end[0]:
                            p_str = 'no answer'
                        else:
                            sss = min(len(dat_list), max(0, start_end[0]))
                            eee = min(len(dat_list), max(0, start_end[1]+1))
                            p_str = tokenizer.decode(dat_list[sss:eee])
                        print ('"{}","{}","{}","{}"'.format(qid, q_str, a_str, p_str), file=f_preds)
        f_preds.flush()
        test_em, test_f1 = sum(total_em) / count, sum(total_f1) / count

        print('-' * 71)
        print('test_em {:7.4f} | test_f1 {:7.4f}'.format(test_em, test_f1))
        print('-' * 71)



if __name__ == '__main__':
    main()
