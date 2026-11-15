# 样片来源与使用条件 / Demo sources and usage conditions

来源核验日期 / Sources checked: 2026-11-15.

## 中文

本说明涵盖本目录原有的 16 个生成 MP4：`t2va-{5,10,15}s.mp4`、`f2va-{5,10,15}s.mp4`、
`fl2va-{5,10,15}s.mp4`、`fl2va-t2va-{5,10,15}s.mp4`、`r2va-{5,10,15}s.mp4`，
以及 `the-spare-moon-30s.mp4`。
本说明也涵盖 `linux/` 子目录中的 Linux 基准测试和功能验证视频。同源角色衍生视频
沿用下述对应的署名、非商业使用及 CC 许可链；自有橙球样例及原创素材在本项目可许可
范围内按 Apache-2.0 提供，不将角色相关权利覆盖为 Apache。所有模型生成视频仍须
遵守适用的模型输出条款。
其中的角色样片是本项目 AI 生成、改编并后期封装的视频，不是 B 站原视频的转存。
文字生成任务使用新编提示词；帧控和参考任务使用 GPT Image 按角色设定重绘的新关键帧或身份图，
再由 H3 生成视频。改编包括新的场景、剧情、镜头和动作，以及 R2VA 中的 Kimi 身份替换。
本说明不将所列创作者或 AI 品牌视为本项目的背书方。

### 角色与来源

角色展示视频 [BV1YZx36WEY8](https://www.bilibili.com/video/BV1YZx36WEY8/)
（汇总发布者 Creeper-北极之光）是寻找造型参考的入口；
该视频发布者不等于下列角色设计的作者。

| 角色 | 设计署名与原始来源 |
| --- | --- |
| DeepSeek 女仆鲸鱼娘 | 上善无形的[鲸鱼娘原型](https://www.bilibili.com/opus/1231977657712771073) → ZipZipPipe 女仆二次设计；[《AI娘化》](https://www.pixiv.net/artworks/148186519)第 1 页（`p0`） |
| Claude | ZipZipPipe；[《AI娘化》](https://www.pixiv.net/artworks/148186519)第 3 页（`p2`） |
| Kimi | ZipZipPipe；[《AI娘化》](https://www.pixiv.net/artworks/148186519)第 6 页（`p5`） |

ZipZipPipe 在[《“亿万”身家的她们》](https://www.bilibili.com/video/BV1bARXBYEZz/)中表示图片开放分享、允许再创作；
其 Pixiv 原图集同时明确要求“需要标注作者，不可商用”。
这不等于对全部角色授予 Apache、CC0 或无限制商用许可。

### 使用条件与改编记录

- **DeepSeek 相关衍生部分**：遵循 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。
  [原型作者声明](https://www.bilibili.com/opus/1231977657712771073)及
  [ZipZipPipe 二创声明](https://www.bilibili.com/opus/1231980045693616162)均确认这条许可链。
  再分享时保留上善无形、ZipZipPipe 的署名、来源和许可链接，说明本项目及自己的修改，
  非商业使用，改编贡献按相同协议分享。本项目在可许可范围内的 DeepSeek 相关改编贡献也按该协议提供。
- **Claude / Kimi 相关素材**：保留 ZipZipPipe 的署名、原作链接，以及作者要求的非商业使用条件。
  未核实作者为这两套设计指定统一的标准 CC 许可证，不将它们套用为 DeepSeek 的 CC BY-NC-SA。
- 本项目的改编记录为：参考角色造型创作新画面、剧情、镜头和动作，部分任务生成关键帧／身份图后制成视频，
  R2VA 参考本项目生成的短剧并替换角色，最后剪辑、封装和压缩。
  这些署名也记录同名样片历史版本的相同来源链；本说明不改写已有归档中的历史文本或重新授权上游作品。

本仓库将样片用于非商业技术展示；免费或开源并不自动意味着任何下游使用都是非商业。
若需商业使用相应角色或视频，应单独处理素材授权，或换用取得相应授权的素材。
本说明只覆盖角色及相关改编条件，不向第三方图像、声音、音乐、名称或商标授予新的权利。

原创代码及通用橙球输入 [`../first.png`](../first.png)、[`../last.png`](../last.png)
仍按仓库 [Apache-2.0](../../LICENSE) 许可；角色和样片条件不限制原创代码的该项商业使用许可。
上游模型输出还须遵守适用的 [MiniMax H3 模型协议](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)，
包括当前第 V.4 节对 Outputs 的地域条件。这份角色说明不替代或解除模型协议义务，
也不声明所有样片可以在全球不受限制地展示或再分发。

## English

This notice covers the 16 original generated MP4s at this directory level:
`t2va-{5,10,15}s.mp4`, `f2va-{5,10,15}s.mp4`, `fl2va-{5,10,15}s.mp4`,
`fl2va-t2va-{5,10,15}s.mp4`, `r2va-{5,10,15}s.mp4`, and
`the-spare-moon-30s.mp4`. It also covers the Linux benchmark and functional proof
videos under `linux/`. Character-derived Linux videos retain the corresponding credits,
non-commercial conditions and CC license chain below. Owned orange-ball examples
and original assets use Apache-2.0 within this project's licensable contributions;
that does not apply Apache terms to character rights. All model-generated videos
remain subject to the applicable model-output terms.

The character samples are AI-generated adaptations packaged by this
project, rather than copies of the original Bilibili video. Text-to-video uses
new prompts; frame-controlled and reference tasks use new GPT Image keyframes
or identity illustrations based on the character designs, followed by H3 video
generation. Changes include new settings, stories, shots and actions, and Kimi
identity replacement in R2VA. Attribution does not imply endorsement by the
creators or AI brands.

### Credits and sources

The character showcase [BV1YZx36WEY8](https://www.bilibili.com/video/BV1YZx36WEY8/)
(showcase publisher: Creeper-北极之光) was a discovery reference.
Its publisher is not the designer of the following characters.

| Character | Design credit and original source |
| --- | --- |
| DeepSeek maid whale-girl | [Original whale-girl by 上善无形](https://www.bilibili.com/opus/1231977657712771073), adapted into the maid design by ZipZipPipe; [AI娘化](https://www.pixiv.net/artworks/148186519), page 1 (`p0`) |
| Claude | ZipZipPipe; [AI娘化](https://www.pixiv.net/artworks/148186519), page 3 (`p2`) |
| Kimi | ZipZipPipe; [AI娘化](https://www.pixiv.net/artworks/148186519), page 6 (`p5`) |

ZipZipPipe permits sharing and further creation in
[this author statement](https://www.bilibili.com/video/BV1bARXBYEZz/).
The author's Pixiv collection also requires attribution and non-commercial use.
This is not a grant of Apache, CC0 or unrestricted commercial rights to every design.

### Conditions and changes

- **DeepSeek-related adaptations** follow
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), as stated
  by the [original creator](https://www.bilibili.com/opus/1231977657712771073) and
  [ZipZipPipe](https://www.bilibili.com/opus/1231980045693616162). Retain both
  creators' names, source and license links; identify this project's and your
  changes; use non-commercially; and share adaptations under the same license.
  This project's licensable DeepSeek-related adaptation contributions are offered
  under that license too.
- **Claude / Kimi assets** retain ZipZipPipe's attribution, source links and
  non-commercial conditions. No uniform standard CC license for these two
  designs was verified; DeepSeek's CC BY-NC-SA must not be assigned to them by assumption.
- Changes made here include new illustrations, stories, shots and actions;
  generated keyframes or identity images for some tasks; character replacement
  using this project's generated short film as the R2VA reference; and editing,
  packaging and compression. These credits also describe the same source chain
  used in historical versions of the named demo files. This notice does not
  rewrite existing archives or relicense upstream works.

The samples are presented here as non-commercial technical demonstrations.
Being free or open source does not make every downstream use non-commercial.
Commercial use of the relevant characters or videos requires separately
addressing asset permissions, or using appropriately licensed replacements.
This notice grants no new rights to third-party images, voices, music, names or trademarks.

Original code and the generic orange-ball inputs
[`../first.png`](../first.png) and [`../last.png`](../last.png) retain the
repository's [Apache-2.0](../../LICENSE) terms; these asset conditions do not
restrict that code license's commercial permissions.
Model outputs remain subject to the applicable
[MiniMax H3 agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
including the current Section V.4 territorial conditions for Outputs. These
character terms do not replace or remove model-agreement obligations, nor claim
that every demo may be displayed or redistributed worldwide without restriction.
