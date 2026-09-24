# CIFAR-10: HLS korak 3 za KV260 / Vitis Unified IDE 2025.2

Ovaj paket sadrži C++ implementaciju točno onog SmallCifarCNN_v1 modela koji je treniran i kvantiziran u prethodnim koracima. Težine i testne primjere preuzima iz TVOG završenog INT8 izvoza. Cilj prve provjere je jednakost svih deset INT32 logita, a ne ponovno mjerenje accuracyja na nekoliko slika.

**1. Raspakiraj sadržaj ZIP-a izravno u postojeći direktorij komponente:**

```text
/home/davorin/Documents/KV260/HLS/cifar10_cnn
```

Datoteke `cnn.cpp`, `cnn.h`, `tb_cnn.cpp` i `prepare_hls.py` trebaju biti uz postojeći `hls_config.cfg`. Paket ne sadrži zamjenski `vitis-comp.json` ni postojeći config; zadržava identitet tvoje komponente.

| Datoteka | Uloga |
|---|---|
| cnn.cpp | Sintetizabilni convolution, ReLU, pooling, GAP i FC; top `cifar10_cnn` |
| cnn.h | Top-level sučelje |
| tb_cnn.cpp | Testbench za C simulaciju i C/RTL cosimulation |
| prepare_hls.py | Provjera i priprema stvarnih težina i golden primjera |

**2. Pripremi svoje težine i testne primjere.** U terminalu s NumPyjem:

```bash
cd /home/davorin/Documents/KV260/HLS/cifar10_cnn
python prepare_hls.py --quant-dir /home/davorin/Documents/KV260/runs/cifar10_01/int8_full
```

Skripta provjerava da se `model_params.h`, `model_int8.npz` i `quantization.json` slažu, provjerava formate i konzervativne granice akumulatora te generira:

- `model_params.h`: tvoje stvarne INT8 težine i INT32 biasi;
- `test_vectors.h`: prva dva dostupna uzorka iz `golden_int8.npz` i njihovi stvarni INT32 izlazi;
- `hls_config.ready.cfg`: konfiguracija s apsolutnim putanjama za tvoju komponentu.

Po potrebi dodaj `--samples 16` za više dostupnih testnih uzoraka. Svaki dodatni uzorak produljuje RTL simulaciju. Testbench nakon uzoraka ponovno pokreće prvi uzorak radi provjere ponovnog korištenja internih buffera.

Skripta ne izračunava nove očekivane logite iz HLS koda: koristi ranije spremljenu integer referencu. `test_vectors.h` pripada samo Test Bench datotekama; testne slike ne sintetiziraju se u CNN IP.

**3. Primijeni konfiguraciju.** Spremi postojeće izmjene u Vitisu, zatim sadržaj generiranog `hls_config.ready.cfg` kopiraj u postojeći `hls_config.cfg`. Možeš to napraviti editorom ili u terminalu iz direktorija komponente:

```bash
cp hls_config.ready.cfg hls_config.cfg
```

Ako Vitis prijavi da je datoteka promijenjena izvana, prihvati ponovno učitavanje sadržaja s diska. Postavke su: K26 dio `xck26-sfvc784-2LV-c`, cilj `vivado`, izvoz `ip_catalog`, takt `10ns`, top funkcija `cifar10_cnn`.

U Sources trebaju biti `cnn.cpp`, `cnn.h` i `model_params.h`; u Test Bench `tb_cnn.cpp` i `test_vectors.h`. Generirani config to već navodi, pa ih nije potrebno još jednom dodavati preko GUI-ja.

**4. Pokreni lijevo FLOW → C SIMULATION → Run.**

Očekivana završna poruka testbencha je:

```text
PASS: all 10 INT32 logits match for every sample and replay.
```

Ta poruka je očekivani kriterij uspjeha, a ne rezultat simulacije na tvojem računalu koji je već poznat. Predikcija ne mora biti jednaka stvarnoj oznaci slike: model ima 72,74% validacijske točnosti. U ovom testu PASS znači da se HLS C++ matematički slaže s integer modelom.

**5. Nakon PASS pokreni C SYNTHESIS → Run.** Pregledaj postignuti/ procijenjeni takt, interval pokretanja, latenciju te DSP, LUT, FF i BRAM procjene. `II=1` u MAC petlji je zahtjev alatu, ne jamstvo da je ostvaren i ne znači jednu cijelu sliku po ciklusu. Ciljanih 100 MHz također treba potvrditi izvještajima.

**6. Nakon uspješne sinteze pokreni C/RTL COSIMULATION → Run.** Isti testbench treba potvrditi sve logite iz generiranog RTL-a. Model ima oko 2,8 milijuna MAC operacija po slici i ova početna arhitektura računa slojeve uzastopno, pa RTL simulacija može trajati znatno dulje od C simulacije.

**7. Nakon RTL PASS pokreni PACKAGE → Run.** Izvezeni Vivado IP zatim dodajemo u Vivado block design. Vivado integracija, bitstream i ARM aplikacija slijede poslije ovih provjera.

## Hardversko sučelje

```cpp
void cifar10_cnn(const uint8_t image[3072], int32_t logits[10]);
```

| Port | Namjena |
|---|---|
| `m_axi_gmem_in` | Čitanje 3072 RGB bajta slike iz DDR-a |
| `m_axi_gmem_out` | Zapis deset INT32 logita u DDR |
| `s_axi_control` | Adrese ulaza/izlaza, start i status |

Ulaz je 32×32 RGB u HWC/interleaved rasporedu. IP preko `INPUT_LUT` odradi istu integer normalizaciju kao Python i podatke pretvori u CHW. ARM aplikacija zato predaje izvorne RGB bajtove, bez dodatne normalizacije. Ulazni i izlazni buffer ne smiju se preklapati. Težine su konstante sintetizirane u IP; promjena težina zahtijeva novu sintezu i bitstream. Točne adrese AXI registara uzimaju se iz generiranog drivera, a ne pretpostavljaju unaprijed. Kod ARM aplikacije treba riješiti vidljivost DDR buffera akceleratoru, uključujući cache održavanje.

## Numerika

- Convolution je cross-correlation s 3×3 kernelom, stride 1 i padding 1.
- INT8 × INT8 daje egzaktan 16-bitni umnožak; akumulacija i bias su INT32.
- ReLU, pomaci, zaokruživanje i saturacija slijede `cnn_requant_relu` iz tvog izvoza.
- MaxPool: 2×2, stride 2. Računaju se četiri convolution piksela po pooled izlazu.
- GAP: `(sum + 8) >> 4`, za 16 nenegativnih vrijednosti.
- FC vraća deset INT32 logita s istom skalom; argmax može izravno na njima.

## Što je provjereno

Isporučeni C++ kod može se provjeriti i običnim C++ compilerom nakon pripreme datoteka:

```bash
g++ -std=c++14 -O2 -Wno-unknown-pragmas cnn.cpp tb_cnn.cpp -o cnn_check
./cnn_check
```

Isporučeni C++ kod provjeren je prema NumPy INT8 referenci na sintetičkim težinama, za pozitivne i negativne kvantizacijske pomake te ponovno pokretanje. Testbench je ispravno odbio namjerno promijenjen očekivani izlaz, a pripremna skripta nepodudarne parametre.

Host C++ test ne izvršava HLS pragme. AMD alati nisu pokrenuti u okruženju autora: uspješnu HLS C simulaciju, sintezu, RTL cosimulation i rad na KV260 treba potvrditi lokalno. Nisu uključeni izmišljeni trenirani parametri ili bitstream.

Reference: [AMD HLS konfiguracijske naredbe](https://docs.amd.com/r/en-US/ug1399-vitis-hls/HLS-Config-File-Commands), [AXI master i kontrolni registar](https://docs.amd.com/r/2022.1-English/ug1399-vitis-hls/Controlling-the-Address-Offset-in-an-AXI4-Interface), [Vivado IP tok](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Target-Flow-Overview).
