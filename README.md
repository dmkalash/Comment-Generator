# Comment-Generator
Russian GPT-3 fine-tuned model for comment prediction on VK post 

Для тестирования необходимо:
Создать файл input.txt, в котором на каждой строке будет располагаться по одному посту.
Рекомендуется приведить текст к нижнему регистру, использовать только символы русского и английского алфавита + пунктуацию.

Ввести следующие команды
chmod +x req.sh
./req.sh
chmod +x run.py
./run.py --file='input.txt' --device='cuda'

Вывод производится в формате(P - post, C - comment):
> P: post
> C: comment
