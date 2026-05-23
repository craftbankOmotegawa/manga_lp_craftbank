#!/usr/bin/env python3
"""
CraftBank 漫画LP画像生成スクリプト
manga_v2.csv の各コマから OpenAI gpt-image-2 で漫画コマ画像を生成する

使い方:
  python3 generate_manga.py --charsheet     # まずキャラデザシートを生成
  python3 generate_manga.py                 # 全コマ生成（キャラシート参照）
  python3 generate_manga.py --start 10      # コマ10から再開
  python3 generate_manga.py --no 5          # コマ5だけ生成
  python3 generate_manga.py --dry-run       # プロンプト確認のみ
"""

import argparse
import base64
import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# .env ファイルからAPIキーを読み込む
load_dotenv(Path(__file__).parent / ".env")

# === 設定 ===
CSV_PATH = Path(__file__).parent / "manga_v2.csv"
OUTPUT_DIR = Path(__file__).parent / "generated_images"
CHARSHEET_PATH = Path(__file__).parent / "generated_images" / "charsheet.png"
MODEL = "gpt-image-2"

# コマサイズ → 画像サイズ・アスペクト比
SIZE_MAP = {
    "大": {"size": "1024x1536", "aspect": "2:3（縦長）"},
    "中": {"size": "1024x1024", "aspect": "1:1（正方形）"},
    "小": {"size": "1024x1024", "aspect": "1:1（正方形）"},
}

# =============================================================
# 画風・キャラクター設定（全プロンプト共通）
# =============================================================
STYLE_RULES = """\
【絶対に守る画風ルール】
・日本の商業青年漫画のフルカラー版と同じ画風
・主線（輪郭線）ははっきりした黒い線。ぼやけない
・着彩はフラットなセル画調（ベタ塗り＋1〜2段階の影）フルカラーで描く
・水彩風、油絵風、グラデーション塗りは禁止
・リアル寄りの大人の体型。頭身は7〜8頭身。デフォルメしすぎない
・表情は自然でリアル。漫画的な過剰演出は避ける
・白黒ではなくフルカラーで描くこと
"""

CHARACTER_GUIDE = """\
【キャラクター設定】
■ 美咲（柏木美咲）— 主人公
・30代後半の日本人女性。経理担当。
・黒髪を低い位置でまとめている。眼鏡をかけている。
・きちんとしたオフィスカジュアル（ブラウス＋カーディガン系）
・真面目で働き者。疲れているが優しい顔。

■ 相川社長
・50代の日本人男性。やや恰幅がいい。
・髪は少し薄くなっている。温厚で父親的な顔。
・スーツ姿。

■ 若林
・20代前半の日本人女性。経理の後輩。
・ショートボブの黒髪。素直で一生懸命な顔。
・オフィスカジュアル。

■ 黒田（工事部長）
・40代の日本人男性。がっしりした体格。日焼けした肌。
・作業着姿。現場たたき上げの雰囲気。

■ 森下（CraftBankコンサルタント）
・30代前半の日本人男性。日焼けした肌。短髪。清潔感。
・ポロシャツにチノパン。スーツではない。
・白いヘルメットを持っていることがある。元施工管理技士。

■ 梶原 — 電話口のみ（顔は描かない）
■ 堀内社長 — 電話口のみ
"""

CHARSHEET_PROMPT = f"""\
キャラクターデザインシートを1枚作ってください。

{STYLE_RULES}
{CHARACTER_GUIDE}

【出力】
以下の5人のキャラクターを1枚の画像にまとめてください:
美咲、相川社長、若林、黒田、森下

それぞれ「正面」「やや斜め」の2ポーズを横に並べる。
背景は白。名前ラベルをキャラの下に入れる。
フキダシやセリフは不要。
全キャラの画風・線の太さ・塗り方を統一すること。
"""


def build_prompt(row: dict) -> str:
    """CSVの1行から画像生成プロンプトを構築する"""
    no = row["No"]
    size_label = row["コマサイズ"]
    speaker = row["話者"]
    dialogue = row["セリフ"].strip().strip('"')
    situation = row["状況描写"].strip().strip('"')
    composition = row["構図・演出"].strip().strip('"')

    size_info = SIZE_MAP.get(size_label, SIZE_MAP["中"])
    aspect = size_info["aspect"]

    lines = [
        "添付のキャラクターデザインシートと同じ画風・同じキャラクターで、漫画の1コマを描いてください。",
        f"アスペクト比: {aspect}",
        "",
        STYLE_RULES,
        CHARACTER_GUIDE,
        f"【コマ番号】{no}（{size_label}コマ）",
        "",
        f"【場面・状況】\n{situation}",
        "",
        f"【構図・演出】\n{composition}",
    ]

    # セリフ → フキダシで描画
    if speaker and speaker not in ("ナレーション", "場面説明", ""):
        lines.append(f"\n【登場人物】{speaker}が中心。")
        if dialogue:
            if dialogue.startswith("（") or dialogue.startswith("("):
                lines.append(
                    f"【フキダシ】雲型の思考フキダシの中に、以下の日本語テキストを正確に書いてください:\n{dialogue}"
                )
            else:
                lines.append(
                    f"【フキダシ】白い楕円のフキダシの中に、以下の日本語テキストを正確に縦書きで書いてください:\n「{dialogue}」"
                )
            lines.append(
                f"【表情】セリフの内容に合った{speaker}の表情と身体の動きを描く。"
            )
    elif speaker == "ナレーション" and dialogue:
        lines.append(
            f"\n【ナレーション】四角いキャプション枠の中に、以下の日本語テキストを正確に書いてください:\n{dialogue}"
        )
    elif speaker == "場面説明":
        lines.append("\n【指示】テキストやフキダシは不要。場面のビジュアルだけ描く。")

    lines.append(
        "\n【重要】フキダシ内の日本語テキストは省略せず、正確に・読みやすく描いてください。"
        "顔や重要な表情にフキダシをかぶせないでください。"
        "添付のキャラデザシートと同じ顔・髪型・服装で描いてください。"
    )

    return "\n".join(lines)


def load_csv() -> list[dict]:
    """CSVファイルを読み込む"""
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def generate_charsheet(client: OpenAI) -> bool:
    """キャラクターデザインシートを生成する。"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if CHARSHEET_PATH.exists():
        print(f"キャラデザシートは既に存在します: {CHARSHEET_PATH}")
        print("再生成する場合は削除してから実行してください。")
        return True

    print("キャラクターデザインシートを生成中... ※60〜120秒かかります")

    try:
        result = client.images.generate(
            model=MODEL,
            prompt=CHARSHEET_PROMPT,
            size="1536x1024",  # 横長でキャラを並べる
            quality="high",
            n=1,
        )

        if result.data[0].b64_json:
            image_bytes = base64.b64decode(result.data[0].b64_json)
            CHARSHEET_PATH.write_bytes(image_bytes)

        print(f"→ 保存完了: {CHARSHEET_PATH}")
        print("このシートを確認し、OKなら各コマの生成に進んでください。")
        return True

    except Exception as e:
        print(f"[ERROR] キャラデザシート生成失敗: {e}")
        return False


def generate_single(client: OpenAI, row: dict, charsheet_b64: str | None,
                    dry_run: bool = False) -> bool:
    """1コマ分の画像を生成して保存する。成功時True。"""
    no = int(row["No"])
    size_label = row["コマサイズ"]
    size_info = SIZE_MAP.get(size_label, SIZE_MAP["中"])
    size = size_info["size"]
    prompt = build_prompt(row)
    output_path = OUTPUT_DIR / f"panel_{no:03d}.png"

    if output_path.exists():
        print(f"  [SKIP] コマ{no}: 既に存在 → {output_path}")
        return True

    if dry_run:
        print(f"\n{'='*60}")
        print(f"コマ{no} ({size_label}, {size})")
        print(f"{'='*60}")
        print(prompt)
        return True

    print(f"  生成中... ({size_label}, {size}) ※30〜90秒かかります")

    try:
        if charsheet_b64:
            # Responses API でキャラデザシートを参照画像として渡す
            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{charsheet_b64}",
                            },
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                tools=[
                    {
                        "type": "image_generation",
                        "quality": "high",
                        "size": size,
                    }
                ],
            )

            # レスポンスから生成画像を取得
            for block in response.output:
                if block.type == "image_generation_call":
                    image_bytes = base64.b64decode(block.result)
                    output_path.write_bytes(image_bytes)
                    break
        else:
            # キャラデザシートなし → Images API で直接生成
            result = client.images.generate(
                model=MODEL,
                prompt=prompt,
                size=size,
                quality="high",
                n=1,
            )

            if result.data[0].b64_json:
                image_bytes = base64.b64decode(result.data[0].b64_json)
                output_path.write_bytes(image_bytes)
            elif result.data[0].url:
                import urllib.request
                urllib.request.urlretrieve(result.data[0].url, output_path)

        print(f"  → 保存完了: {output_path}")
        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="漫画コマ画像生成スクリプト")
    parser.add_argument("--charsheet", action="store_true",
                        help="キャラクターデザインシートを生成")
    parser.add_argument("--start", type=int, default=1, help="開始コマ番号（途中再開用）")
    parser.add_argument("--end", type=int, default=None, help="終了コマ番号")
    parser.add_argument("--no", type=int, default=None, help="特定のコマだけ生成")
    parser.add_argument("--dry-run", action="store_true",
                        help="プロンプト確認のみ（API呼び出しなし）")
    parser.add_argument("--no-ref", action="store_true",
                        help="キャラデザシートを参照せずに生成（Images APIを使用）")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="コマ間の待機秒数（デフォルト: 3秒）")
    args = parser.parse_args()

    # APIキー確認
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEY が設定されていません。")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    if not args.dry_run:
        OUTPUT_DIR.mkdir(exist_ok=True)
        client = OpenAI(timeout=300.0)
    else:
        client = None

    # キャラデザシート生成モード
    if args.charsheet:
        generate_charsheet(client)
        return

    rows = load_csv()
    print(f"CSVから {len(rows)} コマを読み込みました")

    # フィルタリング
    if args.no:
        rows = [r for r in rows if int(r["No"]) == args.no]
    else:
        rows = [r for r in rows if int(r["No"]) >= args.start]
        if args.end:
            rows = [r for r in rows if int(r["No"]) <= args.end]

    if not rows:
        print("対象のコマがありません。")
        sys.exit(0)

    print(f"対象: コマ{rows[0]['No']}〜{rows[-1]['No']} ({len(rows)}コマ)")

    # キャラデザシートの読み込み
    charsheet_b64 = None
    if not args.dry_run and not args.no_ref and CHARSHEET_PATH.exists():
        charsheet_b64 = base64.b64encode(CHARSHEET_PATH.read_bytes()).decode()
        print(f"キャラデザシート参照: {CHARSHEET_PATH}")
    elif not args.dry_run and not args.no_ref:
        print("注意: キャラデザシートがありません。--charsheet で先に生成するか、--no-ref で参照なしで実行してください。")
        print("  → --no-ref で続行します")

    success = 0
    errors = 0

    for i, row in enumerate(rows):
        no = int(row["No"])
        situation = row["状況描写"][:30] + "..." if len(row["状況描写"]) > 30 else row["状況描写"]

        if not args.dry_run:
            print(f"\n[{i+1}/{len(rows)}] コマ{no}: {situation}")

        if generate_single(client, row, charsheet_b64, dry_run=args.dry_run):
            success += 1
        else:
            errors += 1

        # レート制限対策（最後のコマ以外）
        if not args.dry_run and i < len(rows) - 1:
            time.sleep(args.delay)

    if not args.dry_run:
        print(f"\n{'='*40}")
        print(f"完了: 成功 {success} / エラー {errors}")
        print(f"画像フォルダ: {OUTPUT_DIR}/")
        if errors > 0:
            print("エラーが発生したコマは --no オプションで個別に再生成できます。")


if __name__ == "__main__":
    main()
