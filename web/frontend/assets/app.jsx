const { useEffect, useMemo, useRef, useState } = React;

const DATA_TYPE_LABELS = {
  announcement: "지원사업 공고",
  business: "통합공고(사업정보)",
  content: "자료실 콘텐츠",
  statistical: "통계자료",
  lecture: "교육/강좌",
  space: "창업공간",
  center: "창업센터",
  institution: "지원기관",
};

const BASE_PROFILE = {
  name: "창업자",
  age: 29,
  region: "서울",
  business_stage: "예비창업자",
  business_field: "AI",
  target_type: "일반인",
  is_veteran: false,
  is_disabled: false,
  additional_context: "",
  desired_data_types: [],
};

function cleanText(value) {
  return String(value || "")
    .replace(/<\|im_end\|>/g, "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function normalizeUrl(raw) {
  const text = String(raw || "").trim();
  if (!text) return "";
  if (/^https?:\/\//i.test(text)) return text;
  if (text.startsWith("//")) return `https:${text}`;
  if (text.startsWith("www.")) return `https://${text}`;
  return text;
}

function getDataTypeLabel(type) {
  const key = String(type || "").trim().toLowerCase();
  return DATA_TYPE_LABELS[key] || key || "-";
}

function truncateText(value, maxLength = 220) {
  const text = cleanText(value);
  if (!text) return "";
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength).trim()}...`;
}

function getFirstDocLink(doc) {
  const md = doc?.metadata || {};
  const raw = doc?.detail_url || md.detail_url || md.apply_url || md.guide_url || md.biz_aply_url || md.biz_gdnc_url || md.detl_pg_url || md.lctr_pg_url || md.hmpg || "";
  return normalizeUrl(raw);
}

function renderInlineStyledText(text, docs, keyPrefix = "inline") {
  const chunks = String(text || "").split(/(\[\d+\]|\*\*[^*]+\*\*)/g).filter(Boolean);
  return chunks.map((part, idx) => {
    const citationMatch = part.match(/^\[(\d+)\]$/);
    if (citationMatch) {
      const docIndex = Number(citationMatch[1]) - 1;
      const link = Array.isArray(docs) ? getFirstDocLink(docs[docIndex]) : "";
      if (!link) return <React.Fragment key={`${keyPrefix}-c-${idx}`}>{part}</React.Fragment>;
      return <a key={`${keyPrefix}-c-${idx}`} href={link} target="_blank" rel="noreferrer" className="inline-citation">{part}</a>;
    }

    const boldMatch = part.match(/^\*\*([^*]+)\*\*$/);
    if (boldMatch) {
      return <strong key={`${keyPrefix}-b-${idx}`}>{boldMatch[1]}</strong>;
    }

    return <React.Fragment key={`${keyPrefix}-t-${idx}`}>{part}</React.Fragment>;
  });
}

function renderRichAnswer(text, docs) {
  const raw = String(text || "").replace(/\r\n/g, "\n").trim();
  if (!raw) return null;

  const lines = raw.split("\n");
  const blocks = [];
  let bulletBuffer = [];

  const flushBullets = () => {
    if (!bulletBuffer.length) return;
    blocks.push({ type: "list", items: bulletBuffer });
    bulletBuffer = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushBullets();
      continue;
    }

    if (/^[-*•]\s+/.test(trimmed)) {
      bulletBuffer.push(trimmed.replace(/^[-*•]\s+/, ""));
      continue;
    }

    flushBullets();

    const headingMatch = trimmed.match(/^\*\*([^*]+)\*\*$/);
    if (headingMatch) {
      blocks.push({ type: "heading", text: headingMatch[1] });
    } else {
      blocks.push({ type: "paragraph", text: trimmed });
    }
  }

  flushBullets();

  return blocks.map((block, idx) => {
    if (block.type === "heading") {
      return <div key={`block-${idx}`} className="assistant-heading">{renderInlineStyledText(block.text, docs, `h-${idx}`)}</div>;
    }
    if (block.type === "list") {
      return (
        <ul key={`block-${idx}`} className="assistant-list">
          {block.items.map((item, itemIdx) => (
            <li key={`item-${idx}-${itemIdx}`}>{renderInlineStyledText(item, docs, `li-${idx}-${itemIdx}`)}</li>
          ))}
        </ul>
      );
    }
    return <p key={`block-${idx}`} className="assistant-paragraph">{renderInlineStyledText(block.text, docs, `p-${idx}`)}</p>;
  });
}

function ThoughtStream({ lines, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  const items = Array.isArray(lines) ? lines.map(cleanText).filter(Boolean) : [];
  if (!items.length) return null;

  return (
    <div className="thought-stream">
      <div className="thought-stream-header" onClick={() => setOpen((v) => !v)}>
        <div className="thought-stream-label">
          <span className="thought-stream-icon">✦</span>
          처리 과정
          <span className="thought-stream-count">{items.length}</span>
        </div>
        <span className="thought-stream-toggle">{open ? "접기 ▲" : "펼치기 ▼"}</span>
      </div>
      {open && (
        <div className="thought-stream-body">
          {items.map((line, idx) => (
            <div key={idx} className="thought-stream-item">
              <span className="thought-stream-dot" />
              <span className="thought-stream-text">{line}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Sidebar({
  profile,
  setProfile,
  useCache,
  setUseCache,
  runMatchingLoading,
  onRunProfileMatching,
  collapsed,
  onToggleCollapse,
}) {
  const allTypeKeys = Object.keys(DATA_TYPE_LABELS);
  const onProfile = (key, value) => setProfile((prev) => ({ ...prev, [key]: value }));

  const toggleType = (type) => {
    setProfile((prev) => {
      const has = prev.desired_data_types.includes(type);
      return {
        ...prev,
        desired_data_types: has
          ? prev.desired_data_types.filter((x) => x !== type)
          : [...prev.desired_data_types, type],
      };
    });
  };

  const toggleAllTypes = () => {
    setProfile((prev) => {
      const allSelected = prev.desired_data_types.length === allTypeKeys.length;
      return {
        ...prev,
        desired_data_types: allSelected ? [] : [...allTypeKeys],
      };
    });
  };

  return (
    <aside className={`card sidebar-card ${collapsed ? "collapsed" : ""}`}>
      <div className="sidebar-head">
        {!collapsed && <h2>프로필 기반 추천 패널</h2>}
        <button type="button" className="panel-toggle" onClick={onToggleCollapse} title={collapsed ? "프로필 패널 펼치기" : "프로필 패널 접기"}>
          {collapsed ? "▶" : "접기"}
        </button>
      </div>

      {!collapsed && (
        <>
          <p className="muted small">이 패널은 프로필 기반 추천용입니다. 우측 채팅은 항상 함께 사용할 수 있습니다.</p>

          <section className="section-block">
            <h3>기본 프로필</h3>

            <div className="form-row">
              <label>나이</label>
              <input type="number" min="18" max="100" value={profile.age} onChange={(e) => onProfile("age", Number(e.target.value))} />
            </div>

            <div className="form-row">
              <label>지역</label>
              <select value={profile.region} onChange={(e) => onProfile("region", e.target.value)}>
                {["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주", "전국"].map((r) => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
            </div>

            <div className="form-row">
              <label>창업단계</label>
              <select value={profile.business_stage} onChange={(e) => onProfile("business_stage", e.target.value)}>
                {["예비창업자", "1년미만", "2년미만", "3년미만", "5년미만", "7년미만"].map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            <div className="form-row">
              <label>사업분야</label>
              <input value={profile.business_field} onChange={(e) => onProfile("business_field", e.target.value)} />
            </div>

            <div className="form-row">
              <label>대상유형</label>
              <select value={profile.target_type} onChange={(e) => onProfile("target_type", e.target.value)}>
                {["청소년", "대학생", "일반인"].map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
          </section>

          <details className="advanced">
            <summary>조회 데이터 유형</summary>
            <div className="form-row">
              <div className="check-grid">
                <label className="check-item">
                  <input type="checkbox" checked={profile.desired_data_types.length === allTypeKeys.length} onChange={toggleAllTypes} />
                  전체
                </label>
                {allTypeKeys.map((k) => (
                  <label key={k} className="check-item">
                    <input type="checkbox" checked={profile.desired_data_types.includes(k)} onChange={() => toggleType(k)} />
                    {DATA_TYPE_LABELS[k]}
                  </label>
                ))}
              </div>
            </div>
            <label className="check-item" style={{ padding: "0 13px 12px" }}>
              <input type="checkbox" checked={useCache} onChange={(e) => setUseCache(e.target.checked)} />
              캐시 사용
            </label>
          </details>

          <button className="btn btn-block" onClick={onRunProfileMatching} disabled={runMatchingLoading}>
            {runMatchingLoading ? "실행 중..." : "프로필 매칭 실행"}
          </button>
        </>
      )}
    </aside>
  );
}

function ChatPanel({ activeTab, setActiveTab, appMode, setAppMode, messages, input, setInput, onAskQuestion, askLoading, report }) {
  const reportThought = Array.isArray(report?.thought) ? report.thought : [];
  const pickField = (rec, key) => rec?.[key] || rec?.metadata?.[key] || "";

  const looksLikeInternalCode = (value) => {
    const text = cleanText(value);
    if (!text) return false;
    return /^(cmrczn_[A-Za-z0-9]+|[A-Za-z0-9]+_Tab\d+|Tab\d+)$/i.test(text);
  };

  const recommendationDetailRows = (rec) => {
    const md = rec?.metadata || {};
    const dataType = String(rec?.data_type || md?.type || "").toLowerCase();
    const businessCode = md.biz_category_cd || md.field || pickField(rec, "field") || "";
    const compact = (...values) => values.map((v) => cleanText(v)).filter(Boolean).join(" ");

    if (dataType === "announcement" || dataType === "business") {
      return [
        ["지원 대상", pickField(rec, "apply_target") || pickField(rec, "apply_target_desc")],
        ["지원 예산 및 규모", md.biz_supt_bdgt_info || md.support_budget || ""],
        ["지원 내용", md.biz_supt_ctnt || md.support_content || ""],
        ["사업 특성", md.supt_biz_chrct || md.business_characteristics || ""],
        ["사업 소개", md.supt_biz_intrd_info || md.business_intro || rec.summary || ""],
        ["사업 구분 코드", looksLikeInternalCode(businessCode) ? "" : businessCode],
        ["사업 연도", md.biz_yr || md.year || ""],
      ].filter(([, value]) => cleanText(value));
    }

    if (dataType === "content") {
      return [
        ["요약", compact(rec.title, md.fstm_reg_dt || md.reg_date ? `등록일 ${md.fstm_reg_dt || md.reg_date}` : "", "자료입니다.")],
        ["등록일", md.fstm_reg_dt || md.reg_date || ""],
        ["파일명", md.file_nm || md.file_name || ""],
      ].filter(([, value]) => cleanText(value));
    }

    if (dataType === "statistical") {
      return [
        ["요약", compact(rec.title, md.last_mdfcn_dt ? `최종 수정일 ${md.last_mdfcn_dt}` : "", "통계 자료입니다.")],
        ["최초 등록일", md.fstm_reg_dt || md.first_reg_dt || ""],
        ["최종 수정일", md.last_mdfcn_dt || ""],
        ["파일명", md.file_nm || md.file_name || ""],
      ].filter(([, value]) => cleanText(value));
    }

    if (dataType === "lecture") {
      return [
        ["요약", compact(rec.title, cleanText(md.lctr_istc || md.desc) ? `${cleanText(md.lctr_istc || md.desc)}` : "")],
        ["등록일", md.reg_dt || ""],
        ["수정일", md.mdfcn_dt || ""],
        ["키워드", md.kywrd || md.keywords || ""],
      ].filter(([, value]) => cleanText(value));
    }

    if (dataType === "space") {
      return [
        ["요약", compact(md.cntr_nm || md.center_name || "", md.spce_nm || rec.title || "", "공간 정보입니다.")],
        ["공간 유형", md.spce_type_nm || md.space_type || ""],
        ["주소", md.addr || md.address || ""],
        ["예약 가능 여부", md.rsvt_psbl_clss || md.reservation_class || ""],
      ].filter(([, value]) => cleanText(value));
    }

    if (dataType === "center") {
      return [
        ["요약", compact(rec.title, md.regin_clss || md.region ? `${md.regin_clss || md.region} 지역` : "", "센터 정보입니다.")],
        ["센터 유형", md.cntr_type_nm || md.center_type || ""],
        ["지역", md.regin_clss || md.region || ""],
        ["주소", md.addr || md.address || ""],
      ].filter(([, value]) => cleanText(value));
    }

    if (dataType === "institution") {
      return [
        ["요약", compact(rec.title, "지원기관 정보입니다.")],
        ["기관명", md.inst_nm || rec.title || ""],
        ["설립일", md.fndn_dt || ""],
      ].filter(([, value]) => cleanText(value));
    }

    return [
      ["요약", cleanText(rec.summary || "")],
    ].filter(([, value]) => cleanText(value));
  };

  const recommendationViewModel = (rec) => {
    const rows = recommendationDetailRows(rec).filter(([, value]) => cleanText(value));
    const summaryLabels = new Set(["사업 소개", "요약", "지원 내용"]);
    const summaryRow = rows.find(([label]) => summaryLabels.has(label));
    const detailRows = rows.filter(([label]) => !summaryLabels.has(label));
    const highlights = [];

    const pushHighlight = (label, value) => {
      const text = cleanText(value);
      if (!text) return;
      if (highlights.some((item) => item.label === label || item.value === text)) return;
      highlights.push({ label, value: truncateText(text, 80) });
    };

    pushHighlight("지원 대상", rec.apply_target_desc || rec.apply_target);
    pushHighlight("지역", rec.region);
    pushHighlight("주관 기관", rec.host_org);
    pushHighlight("마감", rec.deadline);

    for (const [label, value] of detailRows) {
      if (highlights.length >= 4) break;
      pushHighlight(label, value);
    }

    return {
      summary: cleanText(summaryRow?.[1] || rec.summary || ""),
      highlights,
      detailRows,
    };
  };

  const scoreTone = (score) => {
    const n = Number(score || 0);
    if (n >= 80) return "high";
    if (n >= 50) return "mid";
    return "low";
  };

  const renderStructuredFacts = (facts) => {
    if (!Array.isArray(facts) || !facts.length) return null;
    return (
      <div className="msg-facts">
        <div className="muted small" style={{ marginBottom: 6 }}><strong>참고 정보</strong></div>
        {facts.slice(0, 10).map((pairs, idx) => (
          <div key={idx} className="mini-rec">
            {(Array.isArray(pairs) ? pairs : []).map((pair, j) => {
              const [label, value] = Array.isArray(pair) ? pair : ["", ""];
              if (!label || !value) return null;
              const text = String(value);
              const isUrl = /^https?:\/\//i.test(text);
              return (
                <div key={j} className="muted small">
                  <strong>{label}:</strong>{" "}
                  {isUrl ? <a href={text} target="_blank" rel="noreferrer">{text}</a> : text}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    );
  };

  const firstDocLink = (doc) => {
    const md = doc?.metadata || {};
    const raw = doc?.detail_url || md.detail_url || md.apply_url || md.guide_url || md.biz_aply_url || md.biz_gdnc_url || md.detl_pg_url || md.lctr_pg_url || md.hmpg || "";
    return normalizeUrl(raw);
  };

  const renderSourceDocs = (docs) => {
    if (!Array.isArray(docs) || !docs.length) return null;
    return (
      <div className="assistant-sources">
        <div className="assistant-sources-label">근거 문서</div>
        <div className="assistant-source-grid">
          {docs.slice(0, 4).map((doc, idx) => {
            const md = doc?.metadata || {};
            const title = cleanText(doc?.title || md?.title || `문서 ${idx + 1}`);
            const link = getFirstDocLink(doc);
            const type = getDataTypeLabel(md?.type || doc?.data_type || "");
            const summary = truncateText(doc?.text || md?.cntn || md?.content || "", 140);
            return (
              <div key={`src-${idx}-${title}`} className="assistant-source-card">
                <div className="assistant-source-top">
                  <span className="assistant-source-index">[{idx + 1}]</span>
                  <span className="assistant-source-type">{type}</span>
                </div>
                <div className="assistant-source-title">{title}</div>
                {!!summary && <div className="assistant-source-summary">{summary}</div>}
                {link && <a href={link} target="_blank" rel="noreferrer" className="assistant-source-link">문서 열기</a>}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <main className="card chat-panel">
      <div className="top-tabs">
        <button className={`top-tab ${activeTab === "recommendation" ? "active" : ""}`} onClick={() => setActiveTab("recommendation")}>추천 결과</button>
        <button className={`top-tab ${activeTab === "chat" ? "active" : ""}`} onClick={() => setActiveTab("chat")}>상담 챗봇</button>
      </div>

      {activeTab === "recommendation" && (
        <section className="split-section report-box tab-panel resizable-panel recommendation-panel">
          <div className="panel-head panel-head-stacked">
            <div>
              <h3>프로필 매칭 결과</h3>
              <p className="muted small">프로필 기반 추천 결과는 왼쪽 패널의 조건으로 생성됩니다.</p>
            </div>
            <span className="section-badge">추천 결과</span>
          </div>

          {!report && <p className="muted">아직 프로필 매칭 결과가 없습니다.</p>}

          {report && (
            <>
              <p style={{ margin: "0 0 12px", fontWeight: 700 }}>
                총 추천: <span style={{ color: "var(--brand)" }}>{report.total_matches || 0}</span>건
              </p>

              {(report.recommendations || []).slice(0, 10).map((rec, i) => {
                const view = recommendationViewModel(rec);
                const detailUrl = pickField(rec, "detail_url");
                const applyUrl = pickField(rec, "apply_url");
                return (
                  <article key={rec.id || i} className="result-card">
                    <div className="result-card-head">
                      <div className="result-rank">{rec.rank || i + 1}</div>
                      <div className="result-title-block">
                        <div className="result-title">{rec.title || "제목 없음"}</div>
                        <div className="result-meta-row">
                          <span className="result-type-pill">{getDataTypeLabel(rec?.data_type || rec?.metadata?.type)}</span>
                          <span className={`score-badge ${scoreTone(rec.relevance_score)}`}>연관도 {Number(rec.relevance_score || 0).toFixed(1)}</span>
                          {rec.priority && <span className={`priority-badge ${String(rec.priority).toLowerCase()}`}>{rec.priority}</span>}
                        </div>
                      </div>
                    </div>

                    {!!view.summary && (
                      <div className="result-summary-lead">
                        {truncateText(view.summary, 260)}
                      </div>
                    )}

                    {!!view.highlights.length && (
                      <div className="result-highlight-grid">
                        {view.highlights.map((item) => (
                          <div key={`${item.label}-${item.value}`} className="highlight-card">
                            <span className="highlight-label">{item.label}</span>
                            <span className="highlight-value">{item.value}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {!!(rec.reasons || []).length && (
                      <div className="result-reason-row">
                        <strong>추천 근거</strong>
                        <span>{(rec.reasons || []).map(cleanText).filter(Boolean).join(" · ")}</span>
                      </div>
                    )}

                    {!!view.detailRows.length && (
                      <details className="result-summary">
                        <summary>상세 정보 보기</summary>
                        <div className="result-detail-box">
                          {view.detailRows.map(([label, value]) => (
                            <div key={label} className="result-detail-row muted small">
                              <strong>{label}</strong>
                              <span>{cleanText(value)}</span>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    <div className="result-links small">
                      {detailUrl && <a href={detailUrl} target="_blank" rel="noreferrer">상세 보기</a>}
                      {applyUrl && <a href={applyUrl} target="_blank" rel="noreferrer">신청 바로가기</a>}
                    </div>
                  </article>
                );
              })}

              {reportThought.length > 0 && (
                <div className="thought-box">
                  <ThoughtStream lines={reportThought} defaultOpen={false} />
                </div>
              )}
            </>
          )}
        </section>
      )}

      {activeTab === "chat" && (
        <section className="split-section chat-box tab-panel resizable-panel conversation-panel">
          <div className="panel-head">
            <div>
              <h3>상담 챗봇</h3>
              <p className="muted small">
                {appMode === "general"
                  ? "일반 질문 모드 — 프로필 값이 반영되지 않습니다."
                  : "프로필 반영 모드 — 좌측 프로필 값이 질문 해석에 반영됩니다."}
              </p>
            </div>
            <div className="mode-switch">
              <button className={`pill ${appMode === "general" ? "active" : ""}`} onClick={() => setAppMode("general")}>일반 질문</button>
              <button className={`pill ${appMode === "profile" ? "active" : ""}`} onClick={() => setAppMode("profile")}>프로필 반영</button>
            </div>
          </div>

          <div className="chat-list">
            {messages.length === 0 && <div className="muted" style={{ padding: "12px 0" }}>질문을 입력하고 <strong>질문하기</strong>를 누르세요.</div>}

            {askLoading && (
              <div className="msg bot loading-card">
                <div style={{ color: "var(--muted)", display: "flex", alignItems: "center", gap: 8 }}>
                  <div className="typing-dots"><span /><span /><span /></div>
                  근거 문서를 찾고 있습니다...
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role === "user" ? "user" : "bot"}`}>
                {m.role === "assistant" ? (
                  <div className="assistant-card">
                    <div className="assistant-card-answer">{renderRichAnswer(m.content, m.final_docs)}</div>
                    {renderStructuredFacts(m.structured_facts)}
                    {renderSourceDocs(m.final_docs)}
                    {Array.isArray(m.thought) && m.thought.length > 0 && <ThoughtStream lines={m.thought} defaultOpen={false} />}
                  </div>
                ) : (
                  cleanText(m.content)
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {activeTab === "chat" && (
        <section className="chat-input sticky-input">
          <input value={input} placeholder="질문을 입력하세요. 프로필 없이도 질문 가능합니다." onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && onAskQuestion()} />
          <button className="btn" onClick={onAskQuestion} disabled={askLoading}>{askLoading ? "생각 중..." : "질문하기"}</button>
        </section>
      )}
    </main>
  );
}

function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(340);
  const resizeRef = useRef({ active: false, startX: 0, startWidth: 340 });
  const [profile, setProfile] = useState(BASE_PROFILE);
  const [topN] = useState(10);
  const [useCache, setUseCache] = useState(true);
  const [activeTab, setActiveTab] = useState("recommendation");
  const [appMode, setAppMode] = useState("general");
  const [report, setReport] = useState(null);
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [askLoading, setAskLoading] = useState(false);
  const [runMatchingLoading, setRunMatchingLoading] = useState(false);

  const effectiveProfile = useMemo(() => (appMode === "profile" ? profile : null), [appMode, profile]);

  useEffect(() => {
    const onMove = (event) => {
      if (!resizeRef.current.active) return;
      const delta = event.clientX - resizeRef.current.startX;
      const nextWidth = Math.min(520, Math.max(260, resizeRef.current.startWidth + delta));
      setSidebarWidth(nextWidth);
    };

    const onUp = () => {
      resizeRef.current.active = false;
      document.body.classList.remove("is-resizing");
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  const onStartResize = (event) => {
    if (sidebarCollapsed) return;
    resizeRef.current = { active: true, startX: event.clientX, startWidth: sidebarWidth };
    document.body.classList.add("is-resizing");
  };

  const onAskQuestion = async () => {
    const q = chatInput.trim();
    if (!q || askLoading) return;
    const nextHistory = [...messages, { role: "user", content: q }];
    setMessages(nextHistory);
    setChatInput("");
    setAskLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile: effectiveProfile,
          chat_mode: appMode,
          question: q,
          history: nextHistory,
          mode: "no_intent_dense",
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "채팅 실패");
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: data.answer || "",
        thought: Array.isArray(data.thought) ? data.thought : [],
        final_docs: Array.isArray(data.final_docs) ? data.final_docs : [],
        structured_facts: Array.isArray(data.structured_facts) ? data.structured_facts : [],
      }]);
      setActiveTab("chat");
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", content: `오류: ${err.message}` }]);
    } finally {
      setAskLoading(false);
    }
  };

  const onRunProfileMatching = async () => {
    setRunMatchingLoading(true);
    try {
      const typedQuestion = chatInput.trim();
      const res = await fetch("/api/match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile,
          question: typedQuestion || null,
          top_n: topN,
          mode: "no_intent_dense",
          use_cache: useCache,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "매칭 실패");
      setReport(data);
      setActiveTab("recommendation");
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", content: `오류: ${err.message}` }]);
    } finally {
      setRunMatchingLoading(false);
    }
  };

  return (
    <div className="shell">
      <header className="header">
        <h1>공공 창업 도우미</h1>
      </header>

      <div className={`layout ${sidebarCollapsed ? "sidebar-collapsed" : ""}`} style={{ "--sidebar-width": `${sidebarWidth}px` }}>
        <Sidebar
          profile={profile}
          setProfile={setProfile}
          useCache={useCache}
          setUseCache={setUseCache}
          runMatchingLoading={runMatchingLoading}
          onRunProfileMatching={onRunProfileMatching}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
        />
        <div className="resize-handle" onMouseDown={onStartResize} role="separator" aria-orientation="vertical" aria-label="패널 너비 조절" />
        <ChatPanel
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          appMode={appMode}
          setAppMode={setAppMode}
          messages={messages}
          input={chatInput}
          setInput={setChatInput}
          onAskQuestion={onAskQuestion}
          askLoading={askLoading}
          report={report}
        />
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
