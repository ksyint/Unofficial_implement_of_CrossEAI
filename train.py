import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import get_dataloader
from models.resnet import ResNet50
from models.projection_head import ProjectionHead
from losses.contrastive_loss import ContrastiveLoss

def train(model, projection_head, train_loader, val_loader, criterion_ce, criterion_contrastive, optimizer, num_epochs=25):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        projection_head.train()
        running_loss = 0.0

        for inputs in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            features, outputs = model(inputs)
            projections = projection_head(features)

            loss_ce = criterion_ce(outputs, torch.zeros(outputs.size(0)).long().to(device))  
            loss_contrastive = criterion_contrastive(projections, projections, torch.zeros(outputs.size(0)).long().to(device))  

            loss = loss_ce + loss_contrastive
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
    
        model.eval()
        projection_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)
                features, outputs = model(inputs)
                projections = projection_head(features)
                
                loss_ce = criterion_ce(outputs, torch.zeros(outputs.size(0)).long().to(device))  
                loss_contrastive = criterion_contrastive(projections, projections, torch.zeros(outputs.size(0)).long().to(device))  
                
                loss = loss_ce + loss_contrastive
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best.pth')
            torch.save(projection_head.state_dict(), 'best_project.pth')

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)
    projection_head = ProjectionHead(input_dim=2048, projection_dim=128).to(device)

    train_loader = get_dataloader("traindata", batch_size=1,shuffle=True)
    val_loader = get_dataloader("valdata", batch_size=1, shuffle=False)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_contrastive = ContrastiveLoss()
    optimizer = optim.SGD(list(model.parameters()) + list(projection_head.parameters()), lr=0.001, momentum=0.9)

    train(model, projection_head, train_loader, val_loader, criterion_ce, criterion_contrastive, optimizer)
