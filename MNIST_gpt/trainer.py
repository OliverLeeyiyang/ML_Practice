# trainer.py
import torch
import torchvision
import torchvision.transforms as transforms
from model import EnhancedMNISTModel
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def save_model(model, path='mnist_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(path='mnist_model.pth', device='cpu'):
    model = EnhancedMNISTModel()
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def display_test_samples(model, device, test_loader):
    model.eval()
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        output = model(images)
    
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f'P: {output[i].argmax().item()}, A: {labels[i].item()}')
        plt.axis('off')
    plt.show()

def main():
    have_gpu = torch.backends.mps.is_available()
    device = torch.device("mps") if have_gpu else torch.device("cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = EnhancedMNISTModel().to(device)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, 5):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # After training
    save_model(model, 'mnist_model.pth')
    print('Model saved!')

    # For displaying test samples
    model = load_model('mnist_model.pth', device)
    model.to(device)
    display_test_samples(model, device, test_loader)

if __name__ == '__main__':
    main()
