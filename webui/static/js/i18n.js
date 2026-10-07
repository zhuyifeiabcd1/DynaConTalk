// Tiny i18n: dictionary + data-i18n attributes.  English is the source of truth.
const DICT = {
  en: {
    new_job: "New job", shortcuts: "Keyboard shortcuts", idle: "idle", untitled_job: "Untitled job",
    keyposes: "Keyposes", trajectories: "Trajectories", identities: "Identities", search: "Search", all_speakers: "All speakers",
    no_job_selected: "No job selected", edited: "Edited", source: "Source", toggle_minimap: "Toggle path map", fullscreen: "Fullscreen",
    empty_stage_title: "Drop an audio file to start", empty_stage_sub: "or pick a job from the list below", base_path: "base", scripted_path: "scripted", now: "now",
    play_pause: "Play / pause (Space)", loop: "Loop", insert_keypose: "Insert keypose", add_move: "Add move",
    timeline: "Timeline", fit: "Fit", discard: "Discard", apply_edits: "Apply edits", chunks: "Chunks", audio: "Audio", trajectory: "Trajectory",
    generate: "Generate", edit: "Edit", info: "Info", drop_audio: "Drop audio here or click to browse", reuse_audio: "Reuse the audio of the selected job",
    identity: "Identity", auto: "Auto", base_trajectory: "Base trajectory", choose: "Choose",
    traj_hint: "The base trajectory sets the walking pattern and the chunk length. Add moves on the timeline to script forward steps, turns or a dip.",
    advanced: "Advanced", guidance: "Guidance", render_previews: "Render preview videos", skip_asr: "Skip speech recognition (use the default text)",
    title: "Title", optional: "Optional", generate_motion: "Generate motion", jobs: "Jobs", logs: "Logs", all: "All", active: "Active",
    generations: "Generations", edits: "Edits", failed: "Failed", starred: "Starred", follow: "Follow", drop_to_start: "Drop audio to start a new job",
    select_keypose_first: "Select a keypose in the library first", insert_at: "Insert here", added_move: "Move added", added_keypose: "Keypose inserted",
    nothing_to_apply: "Nothing to apply", applying: "Edit job submitted", queued: "Queued", cancelled: "Cancelled", deleted: "Deleted",
    confirm_delete: "Delete this job and all of its files?", confirm_discard: "Discard all pending edits?", delete: "Delete", cancel: "Cancel", cancel_job: "Cancel job",
    pending_edits: "Pending edits", committed_edits: "Applied edits", no_pending: "Put keyposes or moves on the timeline, then press Apply edits.",
    context_windows: "Regeneration context windows", context_hint: "One window is 56 frames (1.9 s). More windows blend the edit more smoothly into its surroundings, but change a wider span.",
    render: "Render", frame: "Frame", part: "Body part", strength: "Strength", sigma: "Influence σ", bands: "Wavelet bands", set_to_playhead: "Snap to playhead",
    type: "Type", amount: "Amount", steps: "steps", meters: "m", degrees: "deg", start: "Start", end: "End", duration: "Duration", mode: "Mode",
    replace: "Replace base", add: "Add to base", hold: "Hold ratio", remove: "Remove", frames: "frames",
    forward: "Forward", backward: "Backward", strafe_left: "Step left", strafe_right: "Step right", turn_left: "Turn left", turn_right: "Turn right", crouch: "Crouch", rise: "Rise", hold_still: "Stand still",
    job: "Job", state: "State", created: "Created", elapsed: "Elapsed", phase: "Phase", parent: "Source job", lineage: "Edit chain", artifacts: "Artifacts",
    transcript: "Transcript", warnings: "Warnings", checkpoints: "Model", open: "Open", download: "Download", rerender: "Re-render",
    draft: "Draft", draft_hint: "Moves can be scripted on the trajectory lane before generating; keyposes need a generated result first.", no_audio: "Choose an audio file first",
    gpu_busy: "GPU busy", gpu_free: "GPU free", waiting_gpu: "Waiting for GPU",
    rename: "Rename", category: "Category", tags: "Tags", notes: "Notes", save: "Save", saved: "Saved", speaker: "Speaker", time: "Time",
    stature: "Height", shoulders: "Shoulders", sample: "Sample",
    inherited_edits: "Inherited from upstream", from_job: "From job", inherited_hint: "Already baked into this motion — each edit rebuilds on its parent's output.",
    ai_director: "AI director", ai_suggest: "Suggest gestures", ai_thinking: "Thinking…",
    ai_hint: "The model reads the transcript, the word timings and the loudness of each word, then picks poses from the library. Everything it proposes lands in Pending edits for you to review.",
    ai_instruction: "Note to the director", ai_instruction_ph: "Optional: e.g. keep it calm, emphasise the numbers, more pointing…",
    ai_max: "Max gestures", ai_gap: "Min spacing", ai_settings: "API settings", ai_no_key: "Set the API address and key first",
    ai_gap_hint: "{n} frames ≈ {s} s — two gestures closer than this fight over the same regeneration window",
    ai_max_hint: "{d} s of speech — about one every {e} s",
    ai_added: "{n} gestures proposed", ai_none: "The model proposed nothing usable",
    ai_dropped: "Discarded", ai_needs_job: "Select a finished job first", ai_needs_words: "No word timings yet — run ASR for much better placement",
    ai_base_url: "API address", ai_model: "Model", ai_protocol: "Protocol", ai_key: "API key", ai_key_kept: "Leave blank to keep the saved key",
    ai_saved: "Settings saved", ai_endpoint: "Endpoint", ai_by: "Proposed by AI",
    library_empty: "Nothing matches", uncategorized: "Uncategorized", copy_id: "Copy ID",
    tcat_anchored: "Planted", tcat_subtle: "Subtle", tcat_moderate: "Moderate", tcat_active: "Active", tcat_wide: "Wide",
    tcat_anchored_hint: "Stays on the spot", tcat_subtle_hint: "Small shifts of weight", tcat_moderate_hint: "Walks a little",
    tcat_active_hint: "Paces around", tcat_wide_hint: "Covers the stage",
    tfreq_steady: "Steady", tfreq_rhythmic: "Rhythmic", tfreq_restless: "Restless",
    amplitude: "Amplitude", frequency: "Frequency", motion: "Motion", travel_rate: "travel", moves_per_min: "moves/min", moves_row: "moves", pct_moving: "moving",
    default_trajectory: "Default", quiet_default: "Quiet default", turn_rate: "turn",
    cat_point: "Point", cat_open_palms: "Open palms", cat_raise: "Raise", cat_chest: "Chest", cat_count: "Count", cat_fist: "Fist", cat_clasp: "Clasp", cat_side: "Side", cat_rest: "Rest", cat_other: "Other",
    cat_wide: "Arms wide", cat_head: "To face", cat_reach: "Reach out", cat_behind: "Behind back", cat_fold: "Folded", cat_lean: "Leaning",
    part_right_arm: "Right arm", part_left_arm: "Left arm", part_both_arms: "Both arms", part_hands: "Hands", part_torso: "Torso", part_upper_body: "Upper body", part_full_body: "Full body",
    bound_to: "Walks like", borrowed: "borrowed", bound_hint: "Only this speaker\u2019s own trajectories",
    borrow_hint: "Another speaker\u2019s gait \u2014 stride and height will not match",
    traj_mismatch: "Borrowed {a}\u2019s gait ({ca}, {ha}). {b} normally moves {cb} ({hb}) \u2014 the model has not seen this identity walk that way.",
    use_as_base: "Use as base trajectory", set_identity: "Use this identity", path_length: "path", chunk: "Chunk", scripted: "scripted",
    server_error: "Server error", version_line: "Version", disk_free: "Disk free",
    help_space: "Play / pause", help_arrows: "Step 1 frame (Shift: 10)", help_home: "Jump to start / end", help_k: "Insert the selected keypose at the playhead",
    help_t: "Add a move at the playhead", help_del: "Delete selection", help_esc: "Clear selection", help_apply: "Apply edits", help_brackets: "Previous / next chunk", help_f: "Fit timeline",
    words: "Words", transcribe_words: "Align words (Qwen ASR)", transcribing: "Aligning words…", captions: "Captions", word_here: "Word here",
    no_words_hint: "No word timings yet. Press ASR to align every word to frames with Qwen3-ASR + ForcedAligner.", words_queued: "Word alignment queued",
    help_drag: "Drag markers or segments to move them, drag segment edges to resize; keypose cards can be dropped straight onto the timeline.",
    draft_stage_title: "Draft", draft_stage_sub: "Script moves on the trajectory lane, then press Generate motion", generated_with_job: "Moves are submitted with the generation",
    replace_with_selected: "Replace with the selected library keypose", library: "Library", loading: "Loading…",
    just_now: "just now", min_ago: "{n} min ago", h_ago: "{n} h ago",
  },
  zh: {
    new_job: "新建任务", shortcuts: "键盘快捷键", idle: "空闲", untitled_job: "未命名任务",
    keyposes: "关键姿态", trajectories: "轨迹", identities: "说话人", search: "搜索", all_speakers: "全部说话人",
    no_job_selected: "未选择任务", edited: "编辑后", source: "原始", toggle_minimap: "显示/隐藏路径图", fullscreen: "全屏",
    empty_stage_title: "拖入音频开始", empty_stage_sub: "或在下方列表选择一个任务", base_path: "基线", scripted_path: "脚本", now: "当前",
    play_pause: "播放 / 暂停(空格)", loop: "循环", insert_keypose: "插入关键姿态", add_move: "添加动作",
    timeline: "时间轴", fit: "适配", discard: "放弃", apply_edits: "应用编辑", chunks: "分段", audio: "音频", trajectory: "轨迹",
    generate: "生成", edit: "编辑", info: "信息", drop_audio: "拖入音频或点击选择", reuse_audio: "复用当前任务的音频",
    identity: "说话人", auto: "自动", base_trajectory: "基线轨迹", choose: "选择",
    traj_hint: "基线轨迹决定踱步风格和分段长度。在时间轴上添加动作即可脚本化前进、转身或下蹲。",
    advanced: "高级", guidance: "引导强度", render_previews: "渲染预览视频", skip_asr: "跳过语音识别(用默认文本)",
    title: "标题", optional: "可选", generate_motion: "生成动作", jobs: "任务", logs: "日志", all: "全部", active: "进行中",
    generations: "生成", edits: "编辑", failed: "失败", starred: "已收藏", follow: "跟随", drop_to_start: "松开鼠标即新建任务",
    // dynamic
    select_keypose_first: "先在左侧库中选择一个关键姿态", insert_at: "在此帧插入", added_move: "已添加动作", added_keypose: "已插入关键姿态",
    nothing_to_apply: "没有待应用的编辑", applying: "已提交编辑任务", queued: "已加入队列", cancelled: "已取消", deleted: "已删除",
    confirm_delete: "删除这个任务及其全部文件?", confirm_discard: "放弃所有未应用的编辑?", delete: "删除", cancel: "取消", cancel_job: "取消任务",
    pending_edits: "待应用的编辑", committed_edits: "已应用的编辑", no_pending: "把关键姿态或动作放到时间轴上,然后点“应用编辑”。",
    context_windows: "重生成上下文窗口", context_hint: "每个窗口 56 帧(1.9 s)。窗口越多,编辑段与前后衔接越自然,但改动范围越大。",
    render: "渲染", frame: "帧", part: "部位", strength: "强度", sigma: "影响范围 σ", bands: "小波频带", set_to_playhead: "对齐播放头",
    type: "类型", amount: "幅度", steps: "步", meters: "米", degrees: "度", start: "起始", end: "结束", duration: "时长", mode: "模式",
    replace: "替换基线", add: "叠加基线", hold: "停留比例", remove: "移除", frames: "帧",
    forward: "前进", backward: "后退", strafe_left: "左移", strafe_right: "右移", turn_left: "左转", turn_right: "右转", crouch: "下蹲", rise: "抬起", hold_still: "站定",
    job: "任务", state: "状态", created: "创建于", elapsed: "耗时", phase: "阶段", parent: "来源任务", lineage: "编辑链", artifacts: "产物",
    transcript: "语音转写", warnings: "警告", checkpoints: "模型", open: "打开", download: "下载", rerender: "重新渲染",
    draft: "草稿", draft_hint: "生成前可以先在轨迹轨道上写好动作;关键姿态需要先有生成结果。", no_audio: "请先选择音频文件",
    gpu_busy: "GPU 被占用", gpu_free: "GPU 空闲", waiting_gpu: "等待 GPU",
    rename: "重命名", category: "分类", tags: "标签", notes: "备注", save: "保存", saved: "已保存", speaker: "说话人", time: "时间",
    stature: "身高", shoulders: "肩宽", sample: "样本",
    inherited_edits: "上游继承的编辑", from_job: "来自任务", inherited_hint: "已经烘进当前动作里 —— 每次编辑都以父任务的输出为基底重跑。",
    ai_director: "AI 编辑助手", ai_suggest: "生成建议", ai_thinking: "思考中…",
    ai_hint: "模型会读台词、逐词时间轴和每个词的响度,再从库里挑姿态。它提的每一条都进“待应用的编辑”,由你过目后再应用。",
    ai_instruction: "给助手的要求", ai_instruction_ph: "可留空。例如:整体收敛一点、强调数字、多用指点…",
    ai_max: "最多几个", ai_gap: "最小间隔", ai_settings: "接口设置", ai_no_key: "请先填 API 地址和 key",
    ai_gap_hint: "{n} 帧 ≈ {s} 秒 —— 两个手势比这更近会抢同一个重生成窗口",
    ai_max_hint: "整段 {d} 秒 —— 大约每 {e} 秒一个",
    ai_added: "已提出 {n} 个手势", ai_none: "模型没有给出可用的建议",
    ai_dropped: "已丢弃", ai_needs_job: "先选一个已完成的任务", ai_needs_words: "还没有逐词时间轴 —— 先跑 ASR,落点会准很多",
    ai_base_url: "API 地址", ai_model: "模型", ai_protocol: "协议", ai_key: "API key", ai_key_kept: "留空表示不修改已保存的 key",
    ai_saved: "设置已保存", ai_endpoint: "实际请求地址", ai_by: "AI 建议",
    library_empty: "没有匹配的条目", uncategorized: "未分类", copy_id: "复制 ID",
    tcat_anchored: "定点", tcat_subtle: "轻微", tcat_moderate: "适中", tcat_active: "活跃", tcat_wide: "大幅",
    tcat_anchored_hint: "基本不挪动", tcat_subtle_hint: "只有重心微移", tcat_moderate_hint: "小范围走动",
    tcat_active_hint: "来回踱步", tcat_wide_hint: "大范围走动",
    tfreq_steady: "平稳", tfreq_rhythmic: "有节奏", tfreq_restless: "频繁",
    amplitude: "幅度", frequency: "频率", motion: "运动", travel_rate: "位移", moves_per_min: "次/分", moves_row: "移动", pct_moving: "在动",
    default_trajectory: "默认", quiet_default: "安静默认", turn_rate: "转身",
    cat_point: "指点", cat_open_palms: "摊手", cat_raise: "高举", cat_chest: "胸前", cat_count: "数数", cat_fist: "握拳", cat_clasp: "合掌", cat_side: "侧展", cat_rest: "放松", cat_other: "其他",
    cat_wide: "大张", cat_head: "到面部", cat_reach: "前伸", cat_behind: "背手", cat_fold: "抱臂", cat_lean: "俯身",
    part_right_arm: "右臂", part_left_arm: "左臂", part_both_arms: "双臂", part_hands: "双手", part_torso: "躯干", part_upper_body: "上半身", part_full_body: "全身",
    bound_to: "步态来自", borrowed: "借用", bound_hint: "只看这个说话人自己的轨迹",
    borrow_hint: "别人的步态 —— 步幅和身高对不上",
    traj_mismatch: "借用了 {a} 的步态({ca},{ha})。{b} 平时是{cb}({hb}) —— 模型没见过这个身份这样走。",
    use_as_base: "设为基线轨迹", set_identity: "设为说话人", path_length: "路径", chunk: "分段", scripted: "脚本化",
    server_error: "服务器错误", version_line: "版本", disk_free: "磁盘剩余",
    help_space: "播放 / 暂停", help_arrows: "前后 1 帧(Shift:10 帧)", help_home: "跳到开头 / 结尾", help_k: "在播放头插入所选关键姿态",
    help_t: "在播放头添加动作段", help_del: "删除选中项", help_esc: "取消选择", help_apply: "应用编辑", help_brackets: "上一段 / 下一段", help_f: "时间轴适配窗口",
    words: "台词", transcribe_words: "语音转文字对齐(Qwen ASR)", transcribing: "正在对齐台词…", captions: "字幕", word_here: "此处台词",
    no_words_hint: "还没有逐词时间轴。点 ASR 用 Qwen3-ASR + ForcedAligner 把每个词对到帧。", words_queued: "已提交台词对齐任务",
    help_drag: "拖动标记或段落移动位置;拖动段落两端改变长度;把库中的关键姿态卡片直接拖到时间轴上也可以插入。",
    draft_stage_title: "草稿", draft_stage_sub: "先在轨迹轨道上写好动作,再点“生成动作”", generated_with_job: "动作随生成任务一起提交",
    replace_with_selected: "换成库中当前选中的关键姿态", library: "资产库", loading: "加载中…",
    just_now: "刚刚", min_ago: "{n} 分钟前", h_ago: "{n} 小时前",
  },
};

let current = "en";
const listeners = new Set();

export function t(key, vars) {
  let s = (DICT[current] && DICT[current][key]) || DICT.en[key] || fallback(key);
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, v);
  return s;
}

function fallback(key) {
  // turn snake_case keys into readable English when no entry exists
  return key.replace(/^cat_|^part_|^help_/, "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

export function setLang(lang) {
  current = DICT[lang] ? lang : "en";
  try { localStorage.setItem("studio.lang", current); } catch (_) {}
  document.documentElement.lang = current === "zh" ? "zh-CN" : "en";
  applyDom();
  listeners.forEach((fn) => fn(current));
}

export function getLang() { return current; }
export function onLang(fn) { listeners.add(fn); return () => listeners.delete(fn); }

export function applyDom(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
  root.querySelectorAll("[data-i18n-title]").forEach((el) => { el.title = t(el.dataset.i18nTitle); });
}

export function initLang() {
  let saved = null;
  try { saved = localStorage.getItem("studio.lang"); } catch (_) {}
  setLang(saved || ((navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en"));
}
