const ARM_COLORS = {
  'noop': 'fill-blue',
  'toward_healthy': 'fill-green',
  'toward_diverge': 'fill-purple'
};
const CLASS_COLORS = {
  'CYCLE': 'fill-teal',
  'LOOP': 'fill-orange',
  'MUSE': 'fill-purple'
};
const ARM_TARGET = 648;
const SEEDS_TARGET = 24;
const STATE_TARGET = 72;
const RANK_COLOR = { 0: '#3fb950', 1: '#d29922', 2: '#f85149' };

function fmt(n) {
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return String(n);
}

function fmtTime(s) {
  if (s >= 3600) return (s / 3600).toFixed(1) + 'h';
  if (s >= 60) return Math.round(s / 60) + 'm';
  return s + 's';
}

async function load() {
  try {
    const res = await fetch('/assets/ace-status.json?t=' + Date.now());
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const d = await res.json();
    render(d);
  } catch (e) {
    document.getElementById('generated').textContent = 'Error loading: ' + e.message;
  }
}

function render(d) {
  document.getElementById('generated').textContent =
    'Updated ' + new Date(d.generated_at).toLocaleString();

  // Overall
  document.getElementById('total-rows').textContent = d.total_rows;
  document.getElementById('target-rows').textContent = 'target ' + d.target_rows;

  // Count complete triples from by_state
  var totalComplete = 0, totalSeeds = 0;
  for (var s in d.by_state) {
    totalComplete += d.by_state[s].complete_triples || 0;
    totalSeeds += d.by_state[s].seeds_started || 0;
  }
  document.getElementById('complete-triples').textContent = totalComplete;
  document.getElementById('triples-sub').textContent = 'of ' + (27 * SEEDS_TARGET) + ' possible';

  document.getElementById('overall-bar').style.width = d.pct_complete + '%';
  document.getElementById('pct-text').textContent = d.pct_complete + '% complete';

  // GPU status
  var gpu0dot = document.getElementById('gpu0-dot');
  var gpu1dot = document.getElementById('gpu1-dot');
  var gpu0text = document.getElementById('gpu0-text');
  var gpu1text = document.getElementById('gpu1-text');
  if (d.processes && d.processes.active) {
    gpu0dot.className = 'gpu-dot ' + (d.processes.gpu0 ? 'gpu-on' : 'gpu-off');
    gpu1dot.className = 'gpu-dot ' + (d.processes.gpu1 ? 'gpu-on' : 'gpu-off');
    gpu0text.textContent = 'GPU0: ' + (d.processes.gpu0 ? 'running' : 'idle');
    gpu1text.textContent = 'GPU1: ' + (d.processes.gpu1 ? 'running' : 'idle');
  } else {
    gpu0dot.className = 'gpu-dot gpu-off';
    gpu1dot.className = 'gpu-dot gpu-off';
    gpu0text.textContent = 'GPU0: not running';
    gpu1text.textContent = 'GPU1: not running';
  }

  // By class
  var classDiv = document.getElementById('class-bars');
  classDiv.innerHTML = '';
  for (var cls in d.by_class) {
    var c = d.by_class[cls];
    var pct = Math.min(100, (c.rows / c.target) * 100);
    var color = CLASS_COLORS[cls] || 'fill-blue';
    classDiv.innerHTML += '<div style="margin-bottom:10px">' +
      '<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:3px">' +
      '<span>' + cls + ' <span style="color:#8b949e">(' + c.states_started + '/' + c.states + ' states)</span></span>' +
      '<span>' + c.rows + '/' + c.target + '</span>' +
      '</div>' +
      '<div class="bar-track"><div class="bar-fill ' + color + '" style="width:' + pct + '%"></div></div>' +
      '</div>';
  }

  // By arm
  var armDiv = document.getElementById('arm-bars');
  armDiv.innerHTML = '';
  for (var arm in d.by_arm) {
    var a = d.by_arm[arm];
    var pct = Math.min(100, (a.rows / ARM_TARGET) * 100);
    armDiv.innerHTML += '<div style="margin-bottom:10px">' +
      '<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:3px">' +
      '<span>' + arm + '</span>' +
      '<span>' + a.rows + '/' + ARM_TARGET + ' \u00b7 ' + fmtTime(a.avg_elapsed_s) + '/row</span>' +
      '</div>' +
      '<div class="bar-track"><div class="bar-fill ' + (ARM_COLORS[arm] || 'fill-blue') + '" style="width:' + pct + '%"></div></div>' +
      '</div>';
  }

  // By state — only show states with rows > 0, plus a count of pending
  var stateDiv = document.getElementById('state-bars');
  stateDiv.innerHTML = '';
  var activeStates = [];
  var pendingCount = 0;
  for (var s in d.by_state) {
    if (d.by_state[s].rows > 0) {
      activeStates.push(s);
    } else {
      pendingCount++;
    }
  }
  activeStates.sort();
  for (var i = 0; i < activeStates.length; i++) {
    var s = activeStates[i];
    var st = d.by_state[s];
    var pct = Math.min(100, (st.rows / STATE_TARGET) * 100);
    var color = CLASS_COLORS[st.class] || 'fill-blue';
    var badge = '';
    if (st.complete_triples === SEEDS_TARGET) badge = ' <span class="badge badge-cyan">done</span>';
    else if (st.complete_triples > 0) badge = ' <span class="badge badge-yellow">' + st.complete_triples + '/' + SEEDS_TARGET + '</span>';
    else badge = ' <span class="badge badge-purple">in progress</span>';
    stateDiv.innerHTML += '<div style="margin-bottom:6px">' +
      '<div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:2px">' +
      '<span>' + s + badge + '</span>' +
      '<span>' + st.rows + '/' + STATE_TARGET + '</span>' +
      '</div>' +
      '<div class="bar-track"><div class="bar-fill ' + color + '" style="width:' + pct + '%"></div></div>' +
      '</div>';
  }
  if (pendingCount > 0) {
    stateDiv.innerHTML += '<div style="font-size:0.75rem;color:#8b949e;margin-top:8px">' +
      pendingCount + ' states pending \u00b7 ' + (pendingCount * 72) + ' rows queued</div>';
  }

  // Judge results
  var judgeCard = document.getElementById('judge-card');
  if (d.judge) {
    judgeCard.style.display = '';
    var jg = document.getElementById('judge-grid');
    jg.innerHTML = '';
    var arms = ['noop', 'toward_healthy', 'toward_diverge'];
    for (var j = 0; j < arms.length; j++) {
      var arm = arms[j];
      var js = d.judge.arm_stats[arm];
      if (!js) continue;
      var rankColor = RANK_COLOR[js.avg_rank <= 0.85 ? 0 : js.avg_rank <= 1.15 ? 1 : 2];
      var modeHtml = '';
      var modes = js.modes || {};
      var modeOrder = ['progress', 'mixed', 'spinning'];
      for (var k = 0; k < modeOrder.length; k++) {
        var m = modeOrder[k];
        if (modes[m]) {
          modeHtml += '<span class="mode-pill mode-' + m + '">' + m[0].toUpperCase() + modes[m] + '</span>';
        }
      }
      jg.innerHTML += '<div class="judge-arm">' +
        '<div class="judge-arm-name">' + arm.replace('toward_', '→') + '</div>' +
        '<div class="judge-arm-rank" style="color:' + rankColor + '">' + (js.avg_rank !== null ? js.avg_rank.toFixed(2) : '—') + '</div>' +
        '<div class="judge-arm-wins">' + js.wins + ' wins</div>' +
        '<div class="judge-modes">' + modeHtml + '</div>' +
        '</div>';
    }
    document.getElementById('judge-sub').textContent =
      d.judge.pairs_judged + ' pairs judged \u00b7 ' + d.judge.total_verdicts + ' verdicts \u00b7 pilot data only';
  } else {
    judgeCard.style.display = 'none';
  }

  // Run stats
  document.getElementById('window-size').textContent = d.tokens.fixed_window ? d.tokens.fixed_window : '—';
  document.getElementById('total-tokens').textContent = fmt(d.tokens.total);
  document.getElementById('avg-time').textContent = fmtTime(d.elapsed.mean_s) + ' avg/row';
}

load();
setInterval(load, 60000);
