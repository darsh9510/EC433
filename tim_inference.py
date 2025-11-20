import torch
import torch.nn.functional as F
import torch.optim as optim
from model import BaseResNet18
from utils import get_metric_similarity, ALPHA, LAMBDA, TEMP
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TIMSolver:
    def __init__(self, model_path):
        self.model = BaseResNet18(num_classes=64)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(DEVICE)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False

    def get_features(self, support_x, query_x):
        with torch.no_grad():
            _, s_emb = self.model(support_x)
            _, q_emb = self.model(query_x)
        return s_emb, q_emb

    def init_weights(self, support_emb, support_y, n_way):
        weights = []
        for c in range(n_way):
            c_samples = support_emb[support_y == c]
            proto = c_samples.mean(dim=0)
            weights.append(proto)
        return torch.stack(weights).to(DEVICE)
    
    def run_tim_gd(self, support_x, support_y, query_x, n_way, n_query, steps=1000):
        s_emb, q_emb = self.get_features(support_x, query_x)
        W = self.init_weights(s_emb, support_y, n_way)
        W.requires_grad = True
        
        optimizer = optim.Adam([W], lr=1e-3)
        for i in range(steps):
            logits_s = get_metric_similarity(s_emb, W)
            logits_q = get_metric_similarity(q_emb, W)
            ce_loss = F.cross_entropy(logits_s, support_y)
            p_y_given_x = F.softmax(logits_q, dim=1)
            cond_ent = -(p_y_given_x * torch.log(p_y_given_x + 1e-10)).sum(dim=1).mean()
            p_y = p_y_given_x.mean(dim=0)
            marg_ent = -(p_y * torch.log(p_y + 1e-10)).sum()
            loss = LAMBDA * ce_loss + ALPHA * cond_ent - marg_ent
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_logits = get_metric_similarity(q_emb, W)
        return torch.argmax(final_logits, dim=1)
    
    def run_tim_adm(self, support_x, support_y, query_x, n_way, n_query, steps=150):
        s_emb, q_emb = self.get_features(support_x, query_x)
        s_emb = F.normalize(s_emb, p=2, dim=1)
        q_emb = F.normalize(q_emb, p=2, dim=1)
        W = self.init_weights(s_emb, support_y, n_way)
        W = F.normalize(W, p=2, dim=1)
        S_onehot = F.one_hot(support_y, n_way).float()
        
        for i in range(steps):
            logits_q = get_metric_similarity(q_emb, W)
            P = F.softmax(logits_q, dim=1)
            P_sum = P.sum(dim=0, keepdim=True)
            Q = P / torch.sqrt(P_sum + 1e-10)
            Q = Q / Q.sum(dim=1, keepdim=True)
            num_S = torch.mm(S_onehot.t(), s_emb) * LAMBDA
            num_Q = torch.mm(Q.t(), q_emb)
            W_new = num_S + num_Q
            W = F.normalize(W_new, p=2, dim=1)

        final_logits = get_metric_similarity(q_emb, W)
        return torch.argmax(final_logits, dim=1)

def test_pipeline():
    print("Running verification task...")
    solver = TIMSolver("base_model.pth")
    
    n_way = 5
    n_support = 5
    n_query = 15
    
    sx = torch.randn(n_way * n_support, 3, 84, 84).to(DEVICE)
    sy = torch.repeat_interleave(torch.arange(n_way), n_support).to(DEVICE)
    qx = torch.randn(n_way * n_query, 3, 84, 84).to(DEVICE)
    
    print("Testing TIM-GD...")
    pred_gd = solver.run_tim_gd(sx, sy, qx, n_way, n_query)
    print(f"TIM-GD Prediction shape: {pred_gd.shape}")
    
    print("Testing TIM-ADM...")
    pred_adm = solver.run_tim_adm(sx, sy, qx, n_way, n_query)
    print(f"TIM-ADM Prediction shape: {pred_adm.shape}")

if __name__ == "__main__":
    if not os.path.exists("base_model.pth"):
        print("Please run train_base.py first")
    else:
        test_pipeline()