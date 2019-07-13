import torch
from model.optimization import BertAdam

class Trainer:
    def __init__(self, model, learning_rate, train_proc, eval_proc, logger):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = self.get_optimizer()
        self.train_proc = train_proc
        self.eval_proc = eval_proc
        self.logger = logger
        self.best_result = float(1e-5)
        self.global_step = 0

    def get_optimizer(self):
        named_parameters = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        return BertAdam(optimizer_grouped_parameters, lr=self.learning_rate)

    def save_ckpt(self, ckpt_path):
        ckpt = {"global_step": self.global_step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_result":self.best_result}
        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, ckpt_path, map_location=None):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.global_step = ckpt["global_step"]
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_result = ckpt["best_result"]

    def train(self, batch):
        self.model.train()
        loss = self.train_proc.train(self.model, batch)
        loss = loss / float(self.grad_accu_step)
        loss.backward()
        if (self.curr_step+1)%self.grad_accu_step == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.global_step += 1
        return loss.item()

    def start(self, train_data, epoch=1, dev_data=None, valid_every=None, ckpt_path=None, grad_accu_step=1):
        self.curr_step = 0
        self.grad_accu_step = grad_accu_step
        for _ in range(epoch):
            for batch in train_data:
                loss_val = self.train(batch)
                self.curr_step += 1
                info = (loss_val, self.curr_step, len(train_data), self.global_step, self.best_result)
                self.logger.info("Loss: {}, Current Step: {}, Num of Batches {}, Global Step: {}, Best Result: {}...".format(*info))
                if dev_data and self.curr_step > 0 and self.curr_step % valid_every == 0:
                    dev_result = self.test(dev_data)
                    self.logger.info("New Result:{}, Best Result: {}...".format(dev_result, self.best_result))
                    if dev_result > self.best_result:
                        self.best_result = dev_result
                        self.logger.info("New Best Result!!!")
                        if ckpt_path:
                            self.save_ckpt(ckpt_path)
                            self.logger.info("New model saved...")

    def test(self, eval_data):
        self.model.eval()
        self.logger.info("Start evaluation...")
        for bid, batch in enumerate(eval_data):
            self.logger.debug("Evaluating batch {}...".format(bid))
            batch_result = self.eval_proc.eval(self.model, batch)
        return self.eval_proc.aggregate()



