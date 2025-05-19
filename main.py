import torch
import torch.nn.functional as F
import argparse
import numpy as np
import os 
import random
from transformers import AdamW
from transformers import  GPT2Config
from visual_model import CLIPConfig,CLIPEncoder
from tqdm import tqdm
from test_utils import Recorder,top_filtering,compute_scores,get_report
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from model import GPT2LMHeadModel
from tokenizer import Tokenizer
from datasets import IuxrayTrainDataset,MimicTrainDataset,IuxrayTestDataset,MimicTestDataset


def parse_agrs():
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--image_dir', type=str, default='dataset/iu_xray/images/')
    parser.add_argument('--ann_path', type=str, default='dataset/iu_xray/iu_CDRC.json')

    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--max_seq_len', type=int, default=130)
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    parser.add_argument('--epochs', type=int, default=20, help='the number of training epochs.')
    parser.add_argument('--lr', type=float, default=3e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ve', type=float, default=5e-6, help='the learning rate for the visual extractor.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--seed', type=int, default=9233)
    parser.add_argument('--clip_frozen', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=5, help='early stop')
    parser.add_argument('--mode', type=str, default="val", choices=['val', 'test'])
    #test
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=float, default=0, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--no_sample', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='results/', help='address for saving model parameters')
    parser.add_argument('--temp', default=0.2, type=float)
    parser.add_argument('--alpha', default=0.9, type=float)
    parser.add_argument('--beta', default=0.8, type=float)
    parser.add_argument('--gama', default=0.7, type=float)

    args = parser.parse_known_args()[0]
    return args


def change_requires_grad(model, req_grad): 
    for p in model.parameters():
        p.requires_grad = req_grad

def get_optimizer(model, args):
    no_decay = ['bias', 'LayerNorm.weight'] 
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)  
    return optimizer

def save_checkpoint(args, epoch,  model, optimizer,scheduler,best_score,need_update=False):
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        filename = os.path.join(args.save_dir, 'current_checkpoint.pth')
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
            print(f"Created directory {args.save_dir}")
        torch.save(state, filename)
        print("Saving current model...")

        if need_update:
            best_path = os.path.join(args.save_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving best model...")

def save_model_params(model, file_path):
    with open(file_path, 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"Layer: {name}\n")
            f.write(f"Shape: {param.shape}\n")


def main(args):
    random.seed(args.seed)  
    os.environ['PYTHONHASHSEED'] = str(args.seed)    
    np.random.seed(args.seed)   
    torch.manual_seed(args.seed)   
    torch.cuda.manual_seed(args.seed)  
    torch.cuda.manual_seed_all(args.seed)  
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True  
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    # torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset_name=='mimic_cxr':
        args.ann_path='dataset/mimic_cxr/mimic_CDRC.json'
        args.image_dir='dataset/mimic_cxr/images/'
        args.threshold=10
   

    config = CLIPConfig()
    image_encoder = CLIPEncoder(config,device)
    if args.clip_frozen:  
        change_requires_grad(image_encoder, False) 
    else:
        change_requires_grad(image_encoder, True)
    
    tokenizer = Tokenizer(args)
    vocab_size = len(tokenizer.idx2token) 
    special_token_ids = tokenizer.convert_tokens_to_ids(['<unk>', '<bos>', '<eos>', '<pad>','<question>', '<answer>', '<explanation>'])
    
   

    config = GPT2Config(vocab_size=tokenizer.get_vocab_size())
    config.add_cross_attention = True
    config.bos_token_id = special_token_ids[1]
    config.eos_token_id = special_token_ids[2]
    model = GPT2LMHeadModel(config)
    model.set_config(args)
    model.resize_token_embeddings(tokenizer.get_vocab_size())
    model = model.to('cuda')
    print("Model Setup Ready...")

    if args.dataset_name == "iu_xray":
        train_dataset = IuxrayTrainDataset(args, tokenizer, 'train') 
        test_dataset = IuxrayTestDataset(args, tokenizer, 'test')    
        val_dataset = IuxrayTestDataset(args, tokenizer, 'val')
    else:
        train_dataset = MimicTrainDataset(args, tokenizer, 'train')
        test_dataset = MimicTestDataset(args, tokenizer, 'test')
        val_dataset = MimicTestDataset(args, tokenizer, 'val')
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = args.batch_size, shuffle=True, 
                                                num_workers=args.num_workers,pin_memory=False,drop_last=True) 
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 1, shuffle=False, 
                                                num_workers=args.num_workers,pin_memory=False) 
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = 1, shuffle=False, 
                                                num_workers=args.num_workers,pin_memory=True)
   
    optimizer = get_optimizer(model, args)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    metrics = compute_scores

    train_and_val(args,model,image_encoder,train_loader,val_loader,test_loader,optimizer,scheduler,tokenizer,metrics,device)


def train_and_val(args,model,image_encoder,train_loader,val_loader,test_loader,optimizer,scheduler,tokenizer,metrics,device):
    
    recorder = Recorder(args.mode)
    for epoch in range(0, args.epochs):
       
        model.train()
        print(f'start epoch {epoch}')

        train_loss = 0
      
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
           
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            img, input_ids, labels, segment_ids, ques_end, ans_end,cap_end, dec_mask ,ques_mask,answer_idx= batch
          
            if args.dataset_name=='iu_xray':
                att_feats_0 = image_encoder(img[:, 0])
                att_feats_1 = image_encoder(img[:, 1])
                img_embeddings = torch.cat((att_feats_0, att_feats_1), dim=1) 
            else:
                img_embeddings = image_encoder(img)   

            outputs = model(input_ids=input_ids,
                            past_key_values=None,
                            attention_mask=None,
                            token_type_ids=segment_ids,
                            position_ids=None,
                            encoder_hidden_states=img_embeddings,
                            encoder_attention_mask=None,
                            labels=labels,
                            use_cache=False,
                            return_dict=True,
                            ques_end=ques_end,
                            cap_end=cap_end,
                            ans_end=ans_end,
                            dec_mask = dec_mask,
                            ques_mask = ques_mask,
                            answer_idx = answer_idx,
                            is_cam=epoch)
            
            loss = outputs.loss
            train_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log = {'train_loss': train_loss / len(train_loader)} 
        print('\t{:15s}: {}'.format('train_loss', log['train_loss']))


        model.eval()
        image_encoder.eval()
        SPECIAL_TOKENS = ['<unk>', '<bos>', '<eos>', '<pad>','<question>', '<answer>', '<explanation>']
        special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        val_gts, val_res = [], [] 
    
        for i,batch in enumerate(tqdm(val_loader, desc="Valing")):
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            img, input_ids, report_ids, segment_ids = batch
            if args.dataset_name=='iu_xray':
                att_feats_0 = image_encoder(img[:, 0])
                att_feats_1 = image_encoder(img[:, 1])
                img_embeddings = torch.cat((att_feats_0, att_feats_1), dim=1) 
            else:
                img_embeddings = image_encoder(img) 
            current_output = []  
            max_len = args.max_seq_len
            with torch.no_grad():
                for step in range(max_len + 1):
                    if step == max_len: 
                        break
                    outputs = model(input_ids=input_ids, 
                                    past_key_values=None, 
                                    attention_mask=None, 
                                    token_type_ids=segment_ids, 
                                    position_ids=None, 
                                    encoder_hidden_states=img_embeddings, 
                                    encoder_attention_mask=None, 
                                    labels=None, 
                                    use_cache=False, 
                                    return_dict=True)
                    lm_logits = outputs.logits 
                    logits = lm_logits[0, -1, :] / args.temperature 
                    logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)  
                    probs = F.softmax(logits, dim=-1)
                    prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)  
                    if prev.item() in special_tokens_ids:  
                        break 
                    new_segment = special_tokens_ids[-1]  
                    new_segment = torch.LongTensor([new_segment]).to(device)
                    current_output.append(prev.item())   
                    input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)   
                    segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim = 1)
            decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip().lower() 
            generated_report = get_report(decoded_sequences)
            reports = []
            reports.append(generated_report)
            ground_truths = tokenizer.decode_batch(report_ids,skip_special_tokens=True) 
            val_res.extend(reports)
            val_gts.extend(ground_truths)

        model.train()
        val_scores = metrics({i: [gt] for i, gt in enumerate(val_gts)},{i: [re] for i, re in enumerate(val_res)})
        log.update(**{'val_' + k: v for k, v in val_scores.items()})      
        for key, value in val_scores.items():
                print('\t{:15s}: {}'.format('val_' + key, value))
        
        

       
        model.eval()
        image_encoder.eval()
        SPECIAL_TOKENS = ['<unk>', '<bos>', '<eos>', '<pad>','<question>', '<answer>', '<explanation>']
        special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        test_gts, test_res = [], [] 

       
        for i,batch in enumerate(tqdm(test_loader, desc="Testing")):
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            img, input_ids, report_ids, segment_ids = batch
            if args.dataset_name=='iu_xray':
                att_feats_0 = image_encoder(img[:, 0])
                att_feats_1 = image_encoder(img[:, 1])
                img_embeddings = torch.cat((att_feats_0, att_feats_1), dim=1) 
            else:
                img_embeddings = image_encoder(img) 
            current_output = []  
            max_len = args.max_seq_len
            with torch.no_grad():
                for step in range(max_len + 1):
                    if step == max_len: 
                        break
                    outputs = model(input_ids=input_ids, 
                                    past_key_values=None, 
                                    attention_mask=None, 
                                    token_type_ids=segment_ids, 
                                    position_ids=None, 
                                    encoder_hidden_states=img_embeddings, 
                                    encoder_attention_mask=None, 
                                    labels=None, 
                                    use_cache=False, 
                                    return_dict=True)
                    lm_logits = outputs.logits 
                    logits = lm_logits[0, -1, :] / args.temperature 
                    logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)  
                    probs = F.softmax(logits, dim=-1)
                    prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)  
                    if prev.item() in special_tokens_ids:  
                        break 
                    new_segment = special_tokens_ids[-1]  
                    new_segment = torch.LongTensor([new_segment]).to(device)
                    current_output.append(prev.item())   
                    input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)   
                    segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim = 1)
            decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip().lower()
            
            generated_report = get_report(decoded_sequences)
            reports = []
            reports.append(generated_report)
            ground_truths = tokenizer.decode_batch(report_ids,skip_special_tokens=True) 
            
            test_res.extend(reports)
            test_gts.extend(ground_truths)
                
        test_scores = metrics({i: [gt] for i, gt in enumerate(test_gts)},{i: [re] for i, re in enumerate(test_res)})
        log.update(**{'test_' + k: v for k, v in test_scores.items()})      
        
       
        for key, value in test_scores.items():
                print('\t{:15s}: {}'.format('test_' + key, value))

        
        recorder.record_best(log)
        need_update = recorder.needsUpdate(log,epoch)
        save_checkpoint(args,epoch,model,optimizer,scheduler,recorder.best_score,need_update)

        if recorder.check_early_stop(epoch,patience=args.patience):
            print(f'Early stopping triggered! No improvement for {args.patience} epochs.')
            print(f'Best {args.mode}_BLEU4: {recorder.best_score} at epoch {recorder.best_epoch}')
            break

        scheduler.step()


if __name__ == '__main__':

    args = parse_agrs()
    main(args)





