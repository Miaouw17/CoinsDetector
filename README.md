# CoinsDetector

Le projet est séparé en trois dossiers :

- [CoinDetectorRadius](CoinDetectorRadius) : La version final qui détecte les pièces par rayon
- [CoinDetectorAI](CoinDetectorAI) : les tests effectués pour la reconaissance par machine learning
- [CoinDetectorTests](CoinDetectorTests) : Divers test effectués pour divers traitement d'image/ml

## Install
```bash
cd CoinDetectorRadius
pip install -r requirements.txt
```

## Download test files

[https://github.com/Miaouw17/CoinsDetector/wiki/redcoin_example.zip]()

## Example

```bash
cd CoinDetectorRadius
python main.py /path/to/img /path/to/img2 # multiple images
python main.py --debug /path/to/img # enable debug
```
