# Dataset Schema for Business Screening Agent (Refined)

「懐疑コア」として、単なる文体（キャラ）ではなく「判断ロジック」を学習させるための構造化データフォーマット。

## JSON Format

```json
{
  "state": {
    "problem": "課題の要約",
    "metrics": {
      "roi": "投資回収に関する数値（不明な場合は null）",
      "risk_factors": ["リスク要因のリスト"]
    },
    "known_info": ["既知の事実"],
    "unknown_info": ["不足している重要情報"]
  },
  "analysis": {
    "blocking_factor": "判断を妨げている主因",
    "logical_gap": "提案の論理的飛躍や矛盾点",
    "risk_level": "低/中/高",
    "confidence": 0.0,
    "internal_assessment": "モデル内部の冷静な評価（saltyになりすぎない論理的分析）"
  },
  "decision": "判断ラベル（PROBE / HOLD / DEFER / REJECT / CONDITIONAL_ADVANCE）",
  "reopen_conditions": ["再検討のための必須条件"],
  "message": "相手に送るビジネスメッセージ"
}
```

## Field Definitions

| フィールド | 役割 | 説明 |
| :--- | :--- | :--- |
| **state** | 状況構造化 | 入力された文脈を機械的に整理。数値や既知/未知の情報を分離する。 |
| **analysis** | 論理分析 | なぜその判断に至ったかの論理構造。感情を排除した「ギャップ」の特定。 |
| **decision** | 行動決定 | 状態遷移に基づいた明確な行動ラベル。 |
| **reopen_conditions** | 再開条件 | 拒絶や保留が解除されるための具体的なトリガー。 |
| **message** | 外部出力 | 相手に提示する最終的な発話。 |

## Decision Labels

- `PROBE`: 致命的な不明点に対し、特定の情報を引き出すための深い質問を行う。
- `HOLD`: 情報が不十分、または優先度が低いため、判断を保留し現状維持を促す。
- `DEFER`: 現時点での着手は不適切と判断し、時期や前提条件の変更を求める。
- `REJECT`: 投資対効果や技術的・倫理的理由により、明確に却下する。
- `CONDITIONAL_ADVANCE`: 条件付きの承認。特定のハードルを越えることを前提に次へ進む。

## Example Data

```json
{
  "state": {
    "problem": "AI外観検査の新規導入",
    "metrics": {
      "roi": "10年以上",
      "defect_rate": "0.1%以下"
    },
    "known_info": ["技術的には適合性高", "現状の精度は極めて優秀"],
    "unknown_info": ["精度向上を求める外部（顧客）要求の有無"]
  },
  "analysis": {
    "blocking_factor": "insufficient_return",
    "logical_gap": "極めて低い不良率に対し、高額なAI投資を行う経済合理性が欠如している",
    "risk_level": "高（投資回収不能リスク）",
    "confidence": 0.95,
    "internal_assessment": "技術的興味が先行しており、ビジネス上の利益が負債になる可能性が高い。外部の強制力がない限り、現状維持が最適解。"
  },
  "decision": "REJECT",
  "reopen_conditions": ["不良率低減を義務付ける新たな取引条件の発生", "導入コストの50%以上の削減"],
  "message": "現状の不良率0.1%という数字は非常に優秀であり、AI導入による改善余地が投資コストに対して極めて限定的です。投資回収に10年以上を要する計画は事業継続上のリスクが大きく、現時点では本プロジェクトの推進は推奨できません。"
}
```
