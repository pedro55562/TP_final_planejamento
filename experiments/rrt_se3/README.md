# RRT SE(3) Experiment

Este diretório contém o cenário externo usado para testar o RRT bidirecional em
SE(3) no UAV/corpo rígido.

Para rodar a partir da raiz do projeto:

```bash
python3 experiments/rrt_se3/teste_rrt_se3.py
```

Arquivos:

- `teste_rrt_se3.py`: script principal do cenário.
- `setup.py`: helpers de cena e carregamento usados por este experimento.
- `aux_functions.py`: utilidades matemáticas auxiliares do experimento.
- `data/caminho.txt`: caminho/poses de referência usados no cenário.

Por padrão o script não salva HTML. Para salvar, altere `SAVE_ANIMATION` no final
de `teste_rrt_se3.py`; o destino sugerido é `outputs/`.
