# DQN Snake Game

Bu proje, Derin Q-Ağları (DQN) kullanarak eğitilmiş bir yapay zeka ile oynanan bir yılan oyununu içermektedir. Pygame ile geliştirilmiş olan bu oyun, yapay zekanın kendi başına öğrenerek yılanı yönlendirmesini sağlar.

## Kullanılan Teknolojiler

- Python
- Pygame
- PyTorch
- NumPy
- Random
- Collections (deque)

## Nasıl Çalışır?

Bu proje, takviyeli öğrenme (Reinforcement Learning) ile yılanın en iyi hamleleri öğrenmesini amaçlamaktadır. Model, oyun durumu üzerinden tahmin yaparak en uygun hareketi seçer ve deneyim tekrarını kullanarak kendini geliştirir.

### Oyun Mekaniği

- Yılan, 4 yönlü (yukarı, aşağı, sağ, sol) hareket edebilir.
- Yiyecek toplandığında puan kazanılır ve yılan büyür.
- Yılanın kendine çarpması veya duvara çarpması durumunda oyun sona erer.
- Ajan, oyunun mevcut durumunu gözlemleyerek bir hareket seçer ve ödüllendirme mekanizması ile öğrenir.

## Kurulum

Projeyi çalıştırmak için aşağıdaki bağımlılıkları yüklemeniz gerekir:

```bash
pip install pygame torch numpy
```

## Kullanım

Oyunu başlatmak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
python snake.py
```

## Yapay Zeka Modeli

Yapay zeka, basit bir sinir ağı ile eğitilmiştir:

- Giriş katmanı, mevcut oyun durumunu alır (yılanın yönü, yiyeceğin konumu, çarpışma bilgileri).
- İki gizli katman, verileri işleyerek en uygun hareketi belirler.
- Çıkış katmanı, dört olası hareketten birini seçer.

### Ödüllendirme Sistemi

- Yiyecek yendiğinde: **+10 puan**
- Duvara veya kendine çarptığında: **-10 puan**
- Normal hareket: **0 puan**

## Eğitim

Ajanın öğrenme süreci "Deep Q-Network" algoritması ile sağlanmaktadır.

- **Deneyim Tekrarı (Experience Replay)**: Geçmiş deneyimler saklanarak modelin daha verimli öğrenmesi sağlanır.
- **Epsilon-Greedy Politikası**: Rastgele hareketler yaparak keşif ve öğrenme dengesini korur.

## Geliştirme

Eğer projeye katkıda bulunmak isterseniz, aşağıdaki adımları takip edebilirsiniz:

1. Depoyu forklayın.
2. Yeni bir özellik veya iyileştirme ekleyin.
3. Pull request gönderin.

## Lisans

Bu proje MIT Lisansı altında sunulmaktadır.

