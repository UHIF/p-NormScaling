import torch
import torch.nn.functional as F


class pNorm_calibration(torch.nn.Module):
    def __init__(self, model, temperature=1,bias=0.1):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.bias = bias

    def forward(self, input,p_norm):
        # Get the unnormalized logits from the model
        logits = input / (torch.abs(self.temperature)*0.1*torch.norm(input,p=p_norm,dim=1,keepdim=True) + torch.abs(self.bias))

        # Apply the softmax function to get the normalized probabilities
        probabilities = F.softmax(logits, dim=-1)

        return probabilities

    def set_temperature(self, valid_loader,p_norm):
        """
        Tune the temperature using the validation set.
        """
        self.model.eval()

        # Initialize the temperature to 1
        self.temperature=torch.tensor(1).float()
        self.temperature.requires_grad=True
        self.bias = torch.tensor(0.1).float()
        self.bias.requires_grad = True

        optimizer = torch.optim.SGD([self.temperature,self.bias], lr=0.1, momentum=0.9)
        criterion = F.kl_div

        # Optimize the temperature using LBFGS
        for epoch in range(5):
            running_loss = 0.0
            for i, (input, target) in enumerate(valid_loader):
                input = input.cuda()
                target = target.cuda()

                def closure():
                    optimizer.zero_grad()
                    output = self.model(input)
                    probabilities=self.forward(output,p_norm)
                    out_argmax = torch.argmax(probabilities, 1)
                    eval_acc_cls = (out_argmax == target).sum().detach()
                    diff=eval_acc_cls/target.size(0)-torch.mean(torch.logsumexp(probabilities*1000,dim=1)/1000)
                    loss = torch.square(diff)+0.0*criterion(torch.softmax(output,dim=-1).log(),probabilities, reduction='mean')


                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                running_loss += loss.item()

            print('Epoch {}: Loss = {:.4f}'.format(epoch + 1, running_loss / len(valid_loader)))

        self.temperature = self.temperature.detach().cpu().item()
        self.bias = self.bias.detach().cpu().item()

        print('Optimal temperature: {:.5f}'.format(self.temperature))
        print('Optimal bias: {:.5f}'.format(self.bias))

        return abs(self.temperature), abs(self.bias)