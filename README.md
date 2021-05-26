# Comment-Generator
Russian GPT-3 fine-tuned model for comment prediction on VK post\n

Для тестирования необходимо:\n
Создать файл input.txt, в котором на каждой строке будет располагаться по одному посту.\n
Рекомендуется приведить текст к нижнему регистру, использовать только символы русского и английского алфавита + пунктуацию.\n

Ввести следующие команды\n
chmod +x req.sh\n
./req.sh\n
chmod +x run.py\n
./run.py --file='input.txt' --device='cuda'\n

Вывод производится в формате(P - post, C - comment):\n
> P: post\n
> C: comment\n
