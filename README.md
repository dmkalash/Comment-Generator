# Comment-Generator
Russian GPT-3 fine-tuned model for comment prediction on VK post <br />

Для тестирования необходимо: <br />
Создать файл input.txt, в котором на каждой строке будет располагаться по одному посту. <br />
Рекомендуется приведить текст к нижнему регистру, использовать только символы русского и английского алфавита + пунктуацию. <br />

Ввести следующие команды <br />
chmod +x req.sh <br />
./req.sh <br />
chmod +x run.py <br />
./run.py --file='input.txt' --device='cuda' <br />

Вывод производится в формате(P - post, C - comment): <br />
> P: post <br />
> C: comment <br />
