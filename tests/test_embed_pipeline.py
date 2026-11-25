"""Smoke test: AES -> BCH -> model.embed pipeline using the real app code.

Run from the repo root:  python tests/test_embed_pipeline.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

import torch
import numpy as np
from cipher_server import (
    generate_trade_id, 
    encrypt_to_bits, 
    decrypt_from_bits, 
    encode_message_128, 
    decode_message_v2,
    model, 
    device, 
    DEMO_ALPHA,
    TOTAL_PAYLOAD
)

def test_encryption_pipeline():
    print("--- 1. Şifreleme (AES) Testi ---")
    message = "TEST_MESAJI_123"[:16]
    passphrase = "guvenli_sifrem"
    trade_id = generate_trade_id()
    
    # Şifrele (Bit dizisine çevir)
    msg_bits = encrypt_to_bits(message, trade_id, passphrase)
    assert len(msg_bits) == 128, f"Hata: Şifrelenmiş bit uzunluğu 128 olmalı, {len(msg_bits)} bulundu."
    
    # Şifreyi Çöz (Doğrulama)
    decrypted_message = decrypt_from_bits(msg_bits, trade_id, passphrase)
    assert decrypted_message == message, "Hata: Şifre çözme işlemi orijinal mesajı vermedi!"
    print("✔ AES Şifreleme/Çözme başarılı.")

    print("\n--- 2. BCH Hata Düzeltme Kodlaması Testi ---")
    chunks = encode_message_128(msg_bits)
    assert len(chunks) == 4, "Hata: Mesaj 4 chunk'a bölünmeli."
    assert len(chunks[0]) == 63, "Hata: Her bir BCH chunk 63 bit olmalı."
    
    # Hatasız dekodlama testi
    decoded_bits, success_rate = decode_message_v2(chunks)
    assert success_rate == 1.0, "Hata: Hatasız chunk'lar %100 başarıyla çözülmeli."
    assert decoded_bits[:128] == msg_bits, "Hata: BCH dekodlama orijinal bitleri vermedi."
    print("✔ BCH Encode/Decode başarılı.")

    print("\n--- 3. Point Cloud Gömme (Embed) Şekil Testi ---")
    # 4 adet rastgele dummy nokta bulutu (Point Cloud) oluştur
    pc_t = torch.randn(4, 4096, 3, device=device)
    pc_t = pc_t / (pc_t.norm(dim=-1, keepdim=True) + 1e-8)
    
    ch_t = torch.tensor(chunks, dtype=torch.float32, device=device)
    
    model.eval() # Test için modeli değerlendirme moduna al
    with torch.no_grad():
        stego_tensors = model.embed(pc_t, ch_t, DEMO_ALPHA)
        
    assert stego_tensors.shape == (4, 4096, 3), f"Hata: Çıktı boyutu (4, 4096, 3) olmalı, {stego_tensors.shape} bulundu."
    print("✔ Model Embed (Gömme) işlemi hata vermeden çalıştı ve doğru boyutta tensör üretti.")
    
    print("\n🚀 Tüm oluşturma (embed) testleri başarıyla tamamlandı!")

if __name__ == '__main__':
    test_encryption_pipeline()