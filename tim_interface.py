import torch
import torch.optim as optim
import torch.nn.functional as F
from model import ResNet18
from utils import calculate_prototypes, conditional_entropy, marginal_entropy

N_WAY = 5
N_SHOT = 5
N_QUERY = 15
N_TASKS = 100
N_BASE_CLASSES = 64
FEATURE_DIM = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TIM_ALPHA = 0.1
TIM_LAMBDA = 0.1
TIM_ITERATIONS = 1000
TIM_LR = 0.001

def get_stub_task_loader(n_way, n_shot, n_query, n_tasks):
    print("Using STUB task loader for inference...")
    for _ in range(n_tasks):
        n_support = n_way * n_shot
        n_query_total = n_way * n_query
        
        support_images = torch.randn(n_support, 3, 84, 84)
        support_labels = torch.arange(n_way).repeat(n_shot)
        
        query_images = torch.randn(n_query_total, 3, 84, 84)
        query_labels = torch.arange(n_way).repeat(n_query)
        
        yield support_images, support_labels, query_images, query_labels

def main():
    print(f"Starting TIM inference on {DEVICE}...")

    base_model = ResNet18(n_base_classes=N_BASE_CLASSES).to(DEVICE)
    base_model.load_state_dict(torch.load('base_model.pth'))
    
    feature_extractor = base_model.encoder
    feature_extractor.eval()
    
    task_loader = get_stub_task_loader(N_WAY, N_SHOT, N_QUERY, N_TASKS)
    
    total_accuracy = 0
    
    for i, (support_imgs, support_labels, query_imgs, query_labels) in enumerate(task_loader):
        support_imgs = support_imgs.to(DEVICE)
        support_labels = support_labels.to(DEVICE)
        query_imgs = query_imgs.to(DEVICE)
        query_labels = query_labels.to(DEVICE)

        with torch.no_grad():
            support_features = feature_extractor(support_imgs).detach()
            query_features = feature_extractor(query_imgs).detach()
            
        W = calculate_prototypes(support_features, support_labels, N_WAY)
        W.requires_grad_()
        
        classifier_optimizer = optim.Adam([W], lr=TIM_LR)

        for _ in range(TIM_ITERATIONS):
            classifier_optimizer.zero_grad()
            
            support_logits = F.linear(support_features, W)
            loss_ce = F.cross_entropy(support_logits, support_labels)
            
            query_logits = F.linear(query_features, W)
            
            loss_cond_ent = conditional_entropy(query_logits)
            
            loss_marg_ent = marginal_entropy(query_logits)

            loss_total = (TIM_LAMBDA * loss_ce) - loss_marg_ent + (TIM_ALPHA * loss_cond_ent)
            
            loss_total.backward()
            classifier_optimizer.step()

        with torch.no_grad():
            final_logits = F.linear(query_features, W)
            predictions = final_logits.argmax(dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
        
        total_accuracy += accuracy
        
        if (i + 1) % 10 == 0:
            print(f"Task {i+1}/{N_TASKS} | Accuracy: {accuracy*100:.2f}%")

    print(f"\nAverage Accuracy over {N_TASKS} tasks: {total_accuracy / N_TASKS * 100:.2f}%")

if __name__ == '__main__':
    main()