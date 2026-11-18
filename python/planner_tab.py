"""
GUI of the observation planner (v4.37) - built into a frame of the main window.

build(parent, root, vars_dict, hooks) with hooks:
  'to_light_curves'(target)   - set the Light Curves target
  'to_field_variables'(list)  - add targets to the Field Variables list
"""
import datetime as _dt
import os
import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

from python import planner as P
from python import instruments as _inst


def build(parent, root, vars_dict, hooks):
    box = tk.Frame(parent, bg="white")
    box.pack(fill="both", expand=True, padx=20, pady=10)
    tk.Label(box, text="Observation Planner", font=("Segoe UI", 16, "bold"), fg="#1a3a5f", bg="white").pack(anchor="w")
    tk.Label(box, bg="white", fg="#444", font=("Segoe UI", 9), justify="left", wraplength=880,
             text="Which variables can be observed tonight? Targets from the local GCVS list, your priority list or the "
                  "open-cluster list, for the site of an instrument setup, a time window, minimum altitude, azimuth "
                  "sector, distance from the Moon and the magnitude range of the setup. Computed locally - no internet "
                  "needed.").pack(anchor="w", pady=(0, 6))

    def V(key, default, kind=str):
        v = (tk.BooleanVar if kind is bool else tk.StringVar)(value=default)
        vars_dict[key] = (v, kind)
        return v

    # ---------------------------------------------------------------- inputs
    f1 = tk.LabelFrame(box, text="Where and when", bg="white"); f1.pack(fill="x", pady=3)
    setup_var = V("plan_setup", "")
    date_var = tk.StringVar(value=_dt.date.today().isoformat())      # not stored in the session: today
    t0_var = V("plan_from", "dusk"); t1_var = V("plan_to", "dawn")
    dark_var = V("plan_dark", "nautical (-12)")
    tk.Label(f1, text="Instrument setup:", bg="white").grid(row=0, column=0, sticky="w", padx=6, pady=3)
    setup_cb = ttk.Combobox(f1, textvariable=setup_var, width=24, state="readonly")
    setup_cb.grid(row=0, column=1, sticky="w")
    site_lbl = tk.StringVar(value="")
    tk.Label(f1, textvariable=site_lbl, bg="white", fg="#1a3a5f", font=("Segoe UI", 8), wraplength=620,
             justify="left").grid(row=0, column=2, columnspan=6, sticky="w", padx=6)
    tk.Label(f1, text="Night of (date):", bg="white").grid(row=1, column=0, sticky="w", padx=6)
    ttk.Entry(f1, textvariable=date_var, width=12).grid(row=1, column=1, sticky="w")
    tk.Label(f1, text="from", bg="white").grid(row=1, column=2, sticky="e")
    ttk.Entry(f1, textvariable=t0_var, width=7).grid(row=1, column=3, sticky="w", padx=3)
    tk.Label(f1, text="to", bg="white").grid(row=1, column=4, sticky="e")
    ttk.Entry(f1, textvariable=t1_var, width=7).grid(row=1, column=5, sticky="w", padx=3)
    tk.Label(f1, text="(HH:MM local, or dusk / dawn)  Darkness:", bg="white").grid(row=1, column=6, sticky="w")
    ttk.Combobox(f1, textvariable=dark_var, width=15, state="readonly",
                 values=["civil (-6)", "nautical (-12)", "astronomical (-18)", "any (sun ignored)"]).grid(row=1, column=7, sticky="w", padx=3)
    night_lbl = tk.StringVar(value="")
    tk.Label(f1, textvariable=night_lbl, bg="white", fg="#555", font=("Segoe UI", 8), justify="left", wraplength=820).grid(
        row=2, column=0, columnspan=8, sticky="w", padx=6, pady=(0, 3))

    f2 = tk.LabelFrame(box, text="Selection", bg="white"); f2.pack(fill="x", pady=3)
    alt_var = V("plan_alt_min", "30"); sector_var = V("plan_sector", "any"); moon_var = V("plan_moon_min", "20")
    mb_var = V("plan_mag_bright", ""); mf_var = V("plan_mag_faint", "")
    types_var = V("plan_types", ""); pmin_var = V("plan_pmin", ""); pmax_var = V("plan_pmax", ""); amp_var = V("plan_amp_min", "")
    minh_var = V("plan_min_hours", "0.5")
    # v4.39: one frame per line - compact, nothing hidden on small screens
    def line():
        fr = tk.Frame(f2, bg="white"); fr.pack(fill="x", anchor="w", pady=1)
        return fr

    def lab(fr, txt, small=False):
        tk.Label(fr, text=txt, bg="white", fg="#777" if small else "black",
                 font=("Segoe UI", 8) if small else None).pack(side=tk.LEFT, padx=(6, 2))

    def ent(fr, var, w):
        ttk.Entry(fr, textvariable=var, width=w).pack(side=tk.LEFT)
    l1 = line()
    lab(l1, "Altitude >= (deg):"); ent(l1, alt_var, 5)
    # azimuth range in degrees (0 = N, 90 = E, 180 = S, 270 = W); presets fill the fields
    az0_var = V("plan_az_from", "0"); az1_var = V("plan_az_to", "360")
    lab(l1, "   Azimuth (deg) from"); ent(l1, az0_var, 5); lab(l1, "to"); ent(l1, az1_var, 5)
    preset = ttk.Combobox(l1, textvariable=sector_var, width=10, state="readonly", values=list(P.SECTORS.keys()))
    preset.pack(side=tk.LEFT, padx=(6, 0))

    def _preset(*_):
        a0, a1 = P.SECTORS.get(sector_var.get(), (0, 360))
        az0_var.set(f"{a0:g}"); az1_var.set(f"{a1:g}")
    preset.bind("<<ComboboxSelected>>", _preset)
    lab(l1, "0=N 90=E 180=S 270=W; through North e.g. 300 to 60", small=True)
    l2 = line()
    lab(l2, "Moon distance >= (deg):"); ent(l2, moon_var, 5)
    lab(l2, "   Observable at least (h):"); ent(l2, minh_var, 5)
    lab(l2, "   Magnitude from"); ent(l2, mb_var, 5); lab(l2, "to"); ent(l2, mf_var, 5)
    lab(l2, "(empty = limits of the setup)", small=True)
    l3 = line()
    lab(l3, "Types:"); ent(l3, types_var, 16); lab(l3, "e.g. EA EW DSCT* M (empty = all)", small=True)
    lab(l3, "   Period (d) from"); ent(l3, pmin_var, 6); lab(l3, "to"); ent(l3, pmax_var, 6)
    lab(l3, "   Amplitude >="); ent(l3, amp_var, 5)

    f3 = tk.Frame(box, bg="white"); f3.pack(fill="x", pady=4)
    source_var = V("plan_source", "gcvs")
    for txt, val in (("Local GCVS list", "gcvs"), ("Priority list", "priority"), ("Open clusters", "clusters")):
        ttk.Radiobutton(f3, text=txt, variable=source_var, value=val).pack(side=tk.LEFT, padx=6)
    find_btn = tk.Button(f3, text="Find Targets", bg="#388e3c", fg="white", font=("Arial", 10, "bold"), padx=14)
    find_btn.pack(side=tk.LEFT, padx=12)
    status = tk.StringVar(value="")
    tk.Label(f3, textvariable=status, bg="white", fg="#1a3a5f", font=("Segoe UI", 9), wraplength=520, justify="left").pack(side=tk.LEFT)

    # ---------------------------------------------------------------- results
    cols = ("name", "ra", "dec", "type", "mag", "period", "window", "best", "alt", "az", "transit", "hours", "moon", "note")
    heads = {"name": "Name", "ra": "RA", "dec": "Dec", "type": "Type", "mag": "Mag range",
             "period": "Period (d)", "window": "Observable", "best": "Best", "alt": "Alt (deg)",
             "az": "Az (deg)", "transit": "Meridian", "hours": "Hours", "moon": "Moon (deg)", "note": "Note"}
    widths = {"name": 110, "ra": 80, "dec": 60, "type": 65, "mag": 95, "period": 75, "window": 115, "best": 50,
              "alt": 75, "az": 70, "transit": 65, "hours": 50, "moon": 75, "note": 200}
    tf = tk.Frame(box, bg="white"); tf.pack(fill="both", expand=True)
    tree = ttk.Treeview(tf, columns=cols, show="headings", height=14, selectmode="extended")
    for c in cols:
        tree.heading(c, text=heads[c], command=lambda c=c: sort_by(c))
        tree.column(c, width=widths[c], anchor="w" if c in ("name", "note", "type") else "center")
    sb = ttk.Scrollbar(tf, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side=tk.LEFT, fill="both", expand=True); sb.pack(side=tk.LEFT, fill="y")

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    fig = Figure(figsize=(9, 2.6), dpi=90)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.2)
    canvas = FigureCanvasTkAgg(fig, master=box)
    canvas.get_tk_widget().pack(fill="x", pady=4)

    bf = tk.Frame(box, bg="white"); bf.pack(fill="x", pady=(4, 0))
    bf2 = tk.Frame(box, bg="white"); bf2.pack(fill="x", pady=(0, 4))
    state = {'rows': [], 'times': None, 'night': None, 'sort': ('hours', True), 'alt_min': 30.0, 'items': {},
             'catalog': []}

    # ---------------------------------------------------------------- helpers
    def setups_refresh(*_):
        names = _inst.list_names()
        setup_cb['values'] = names
        act = vars_dict.get('active_instrument', (tk.StringVar(value=""),))[0].get()
        if setup_var.get() not in names:
            setup_var.set(act if act in names else (names[0] if names else ""))
        s = _inst.load(setup_var.get()) if setup_var.get() else None
        if s:
            site = s.get('site') or {}
            sett = s.get('settings') or {}
            lat = site.get('lat') if site.get('lat') is not None else sett.get('ts_lat')
            lon = site.get('lon') if site.get('lon') is not None else sett.get('ts_lon')
            pl = s.get('planning') or {}
            site_lbl.set(f"Site: {site.get('name') or '?'}  lat {lat}  lon {lon}  time zone {site.get('timezone') or '(computer)'}"
                         f"   |   magnitudes {pl.get('mag_bright') if pl.get('mag_bright') is not None else '-'} .. "
                         f"{pl.get('mag_faint') if pl.get('mag_faint') is not None else '-'}   |   {_inst.summary(s)}")
        else:
            site_lbl.set("No instrument setup - site from Settings (Site Latitude/Longitude).")
    setup_cb.bind("<<ComboboxSelected>>", setups_refresh)
    vars_dict['planner_refresh'] = setups_refresh

    def site_and_limits():
        s = _inst.load(setup_var.get()) if setup_var.get() else None
        site = (s or {}).get('site') or {}
        sett = (s or {}).get('settings') or {}
        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        lat = num(site.get('lat')) if site.get('lat') is not None else num(sett.get('ts_lat'))
        lon = num(site.get('lon')) if site.get('lon') is not None else num(sett.get('ts_lon'))
        if lat is None or lon is None:
            try:
                lat = float(vars_dict['ts_lat'][0].get()); lon = float(vars_dict['ts_lon'][0].get())
            except Exception:
                raise ValueError("No site coordinates - enter them in the instrument setup (Manage Setups...).")
        pl = (s or {}).get('planning') or {}
        fov = _inst.derived(s) if s else {}
        return lat, lon, P.get_tz(site.get('timezone')), num(pl.get('mag_bright')), num(pl.get('mag_faint')), fov

    def numv(var):
        t = str(var.get()).strip().replace(',', '.')
        return float(t) if t else None

    def fmt_t(t):
        return t.strftime("%H:%M") if t else ""

    def ra_txt(v):
        if v is None:
            return ""
        h = (float(v) % 360) / 15.0
        hh = int(h); mm = (h - hh) * 60
        return f"{hh:02d}:{mm:04.1f}"

    def dec_txt(v):
        if v is None:
            return ""
        d = abs(float(v)); dd = int(d); mm = int(round((d - dd) * 60))
        if mm == 60:
            dd += 1; mm = 0
        return f"{'-' if float(v) < 0 else '+'}{dd:02d}:{mm:02d}"

    def mag_text(t):
        mx, mn = P._f(t.get('max')), P._f(t.get('min'))
        if mx is None and mn is None:
            return ""
        return f"{mx:.1f}-{mn:.1f}" if (mx is not None and mn is not None) else f"{(mx if mx is not None else mn):.1f}"

    def fill_tree():
        tree.delete(*tree.get_children())
        state['items'] = {}
        key, rev = state['sort']
        def sk(r):
            t = r['target']
            return {'name': P.name_sort_key(t.get('name', '')), 'ra': float(t.get('ra') or 0),
                    'dec': float(t.get('dec') or 0), 'az': r.get('best_az', -1) if r.get('best_az') is not None else -1,
                    'transit': r['transit'].timestamp() if r.get('transit') else 0,
                    'type': str(t.get('type') or ''),
                    'mag': P._f(t.get('max')) or 99, 'period': P._f(t.get('period')) or 1e9,
                    'window': r['first'].timestamp() if r.get('first') else 0, 'best': r['best'].timestamp() if r.get('best') else 0,
                    'alt': r.get('best_alt', -99), 'hours': r.get('hours', 0), 'moon': r.get('moon_sep', 0),
                    'note': r.get('note', '')}[key]
        for r in sorted(state['rows'], key=sk, reverse=rev):
            t = r['target']
            p = P._f(t.get('period'))
            vals = (t.get('name', ''), ra_txt(t.get('ra')), dec_txt(t.get('dec')), t.get('type') or '', mag_text(t),
                    "" if p is None else f"{p:.5g}",
                    f"{fmt_t(r.get('first'))}-{fmt_t(r.get('last'))}" if r.get('hours') else "",
                    fmt_t(r.get('best')) if r.get('hours') else "",
                    f"{r['best_alt']:.0f}" if r.get('best_alt') is not None else "",
                    f"{r['best_az']:.0f}" if r.get('best_az') is not None else "",
                    fmt_t(r.get('transit')) + ("" if r.get('transit_in_window', True) else "*") if r.get('transit') else "",
                    f"{r.get('hours', 0):.1f}", f"{r['moon_sep']:.0f}" if r.get('moon_sep') is not None else "",
                    r.get('note', ''))
            iid = tree.insert("", tk.END, values=vals)
            state['items'][iid] = r

    def sort_by(c):
        k, rev = state['sort']
        state['sort'] = (c, not rev if k == c else c in ('hours', 'alt', 'moon'))
        for c_ in cols:                                   # arrow on the sorted column
            tree.heading(c_, text=heads[c_] + ((" v" if state['sort'][1] else " ^") if c_ == c else ""))
        fill_tree()

    def draw(sel_rows=None):
        ax.clear()
        times, night = state['times'], state['night']
        if times is None:
            canvas.draw_idle(); return
        x = np.array([(t - times[0]).total_seconds() / 3600.0 for t in times])
        dark = night['dark']
        ax.fill_between(x, 0, 90, where=~dark, color="#dde3ea", step="mid", label="not dark")
        ax.axhline(state['alt_min'], color="#c62828", lw=0.8, ls="--")
        ax.plot(x, night['moon_alt'], color="#999", lw=1, ls=":", label=f"Moon ({night['moon_illum'] * 100:.0f} %)")
        rows = sel_rows if sel_rows else state['rows'][:6]
        for r in rows:
            if r.get('alt') is None:
                continue
            ln, = ax.plot(x, r['alt'], lw=1.0)
            ok = r.get('ok')
            if ok is not None and ok.any():
                ax.plot(x[ok], r['alt'][ok], lw=2.5, color=ln.get_color(), label=r['target'].get('name', ''))
        title_extra = ""
        if len([r for r in rows if r.get('alt') is not None]) == 1:
            r = next(r for r in rows if r.get('alt') is not None)
            if r.get('transit'):
                xt = (r['transit'] - times[0]).total_seconds() / 3600.0
                if x[0] <= xt <= x[-1]:
                    ax.axvline(xt, color="#333", lw=1.0, ls="--")
                    ax.text(xt, 2, f" meridian {r['transit'].strftime('%H:%M')}", fontsize=7, rotation=90, va="bottom")
                else:
                    title_extra = f" - meridian passage {r['transit'].strftime('%H:%M')} (outside the window)"
        off = (60 - times[0].minute) % 60 / 60.0 - times[0].second / 3600.0   # first full hour
        ticks = np.arange(off, x[-1] + 0.01, 1.0 if x[-1] <= 12 else 2.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels([(times[0] + _dt.timedelta(hours=float(h))).strftime("%H:%M") for h in ticks], fontsize=7)
        ax.set_ylim(0, 90); ax.set_xlim(x[0], x[-1])
        ax.set_ylabel("Altitude (deg)", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_title("Altitude (thick: observable in the chosen window/sector)" + title_extra, fontsize=8)
        ax.legend(fontsize=6, loc="upper right", ncol=4)
        canvas.draw_idle()

    def selected_rows():
        return [state['items'][i] for i in tree.selection() if i in state['items']]
    tree.bind("<<TreeviewSelect>>", lambda e: draw(selected_rows()))

    # ---------------------------------------------------------------- search
    def find():
        try:
            lat, lon, tz, sb_, sf_, fov = site_and_limits()
            date = _dt.date.fromisoformat(date_var.get().strip())
            dark_alt = {"civil": -6.0, "nautical": -12.0, "astronomical": -18.0}.get(dark_var.get().split()[0], 90.0)
            ev, mo = P.twilight_times(date, tz, lat, lon, -12.0 if dark_alt > 0 else dark_alt)
            t0, t1 = t0_var.get().strip().lower(), t1_var.get().strip().lower()
            if t0 == 'dusk':
                t0 = ev.strftime("%H:%M") if ev else "18:00"
            if t1 == 'dawn':
                t1 = mo.strftime("%H:%M") if mo else "06:00"
            times, jd = P.time_grid(date, t0, t1, tz, step_min=10)
            alt_min = numv(alt_var) or 0.0
            a0_, a1_ = float(str(az0_var.get()).replace(',', '.')), float(str(az1_var.get()).replace(',', '.'))
            sector = (0, 360) if (a0_ % 360 == a1_ % 360 or (a0_ <= 0 and a1_ >= 360)) else (a0_ % 360, a1_ % 360 if a1_ != 360 else 360.0)
            moon_min = numv(moon_var)
            bright = numv(mb_var) if str(mb_var.get()).strip() else sb_
            faint = numv(mf_var) if str(mf_var.get()).strip() else sf_
            types = [x for x in str(types_var.get()).replace(',', ' ').split() if x]
            pmin, pmax, amp = numv(pmin_var), numv(pmax_var), numv(amp_var)
            min_h = numv(minh_var) or 0.0
        except Exception as e:
            messagebox.showerror("Observation Planner", str(e)); return
        src = source_var.get()
        find_btn.config(state=tk.DISABLED)
        status.set("Computing ...")
        state['alt_min'] = alt_min

        def worker():
            try:
                night = P.night_info(times, jd, lat, lon, dark_alt if dark_alt < 0 else 90.0)
                note_extra = ""
                if src == "gcvs":
                    cat = P.gcvs_targets()
                    if not cat:
                        raise ValueError("No local GCVS list found - set the local catalogue folder in Settings "
                                         "(a file with 'gcvs' in its name).")
                    tg = P.filter_targets(cat, bright, faint, types, pmin, pmax, amp)
                    rows, _ = P.plan(tg, times, jd, lat, lon, alt_min, sector, moon_min, night=night)
                    rows = [r for r in rows if r['hours'] >= max(min_h, 1e-9)]
                    for r in rows:
                        r['note'] = ""
                    note_extra = f"{len(cat)} stars in the GCVS list, {len(tg)} after the type/period/magnitude filters"
                elif src == "priority":
                    tg = P.load_priority()
                    if not tg:
                        raise ValueError("The priority list is empty - add targets with 'Add to Priority List'.")
                    rows, _ = P.plan(tg, times, jd, lat, lon, alt_min, sector, moon_min, night=night)
                    seen = {id(r['target']) for r in rows}
                    for r in rows:
                        if r['hours'] > 0:
                            r['note'] = "observable in your window"
                        else:
                            r['note'] = (f"above {alt_min:g} deg elsewhere: best {r['best'].strftime('%H:%M')} "
                                         f"alt {r['best_alt']:.0f} az {r['best_az']:.0f}")
                        if r['target'].get('notes'):
                            r['note'] += " | " + r['target']['notes']
                    for t in tg:
                        if id(t) not in seen:
                            rows.append({'target': t, 'hours': 0.0, 'note': "not observable in this night"
                                         + (" | " + t['notes'] if t.get('notes') else ""), 'alt': None})
                    note_extra = f"{len(tg)} targets in the priority list"
                else:
                    cl = P.load_clusters()
                    cat = P.gcvs_targets()
                    fr = None
                    if fov and fov.get('fov_y_deg'):
                        fr = fov['fov_y_deg'] * 60 / 2
                    tg = []
                    for c in cl:
                        vs = P.cluster_variables(c, cat, field_radius_arcmin=fr) if cat else []
                        vs = P.filter_targets(vs, bright, faint, types, pmin, pmax, amp) if vs else []
                        tg.append({'name': c['name'], 'ra': c['ra'], 'dec': c['dec'], 'type': 'cluster',
                                   '_alias': c['alias'], '_vars': vs, '_radius': c['radius_arcmin']})
                    rows, _ = P.plan(tg, times, jd, lat, lon, alt_min, sector, moon_min, night=night)
                    seen = {id(r['target']) for r in rows}
                    for r in rows:
                        t = r['target']
                        r['note'] = (f"{t['_alias']} - {len(t['_vars'])} known variable(s) in the field"
                                     + ("" if r['hours'] > 0 else " - above the limit only outside your sector"))
                    for t in tg:
                        if id(t) not in seen:
                            rows.append({'target': t, 'hours': 0.0, 'note': f"{t['_alias']} - not observable", 'alt': None})
                    note_extra = (f"{len(cl)} clusters (config/open_clusters.csv in the user folder); variables from the "
                                  "local GCVS list" + ("" if cat else " - no local GCVS list found"))
                state.update(rows=rows, times=times, night=night, jd=jd)

                def done():
                    fill_tree(); draw()
                    n_ok = sum(1 for r in rows if r.get('hours', 0) > 0)
                    status.set(f"{n_ok} target(s) observable ({note_extra}).")
                    night_lbl.set(f"Night {date.isoformat()}: window {times[0].strftime('%H:%M')}-{times[-1].strftime('%H:%M')} "
                                  f"local; darkness (sun < -12 deg) {fmt_t(ev) or 'never'} - {fmt_t(mo) or ''}; "
                                  f"Moon {night['moon_illum'] * 100:.0f} % illuminated, up "
                                  f"{100 * np.mean(night['moon_alt'] > 0):.0f} % of the window. Site {lat:.2f}, {lon:.2f}.")
                root.after(0, done)
            except Exception as e:
                root.after(0, lambda err=e: (status.set(f"Error: {err}"), messagebox.showerror("Observation Planner", str(err))))
            finally:
                root.after(0, lambda: find_btn.config(state=tk.NORMAL))
        threading.Thread(target=worker, daemon=True).start()
    find_btn.config(command=find)

    # ---------------------------------------------------------------- actions
    def sel_targets(expand_clusters=True):
        out = []
        for r in selected_rows():
            t = r['target']
            if t.get('type') == 'cluster' and expand_clusters:
                out += [dict(v) for v in t.get('_vars', [])]
            else:
                out.append({k: v for k, v in t.items() if not k.startswith('_')})
        return out

    def to_priority():
        ts = [t for t in sel_targets(expand_clusters=False)]
        if not ts:
            messagebox.showinfo("Observation Planner", "Select targets in the list first."); return
        n = P.add_priority(ts)
        status.set(f"{n} target(s) added to the priority list ({P.PRIORITY_FILE}).")

    def from_priority():
        names = [r['target'].get('name', '') for r in selected_rows()]
        if not names:
            return
        if source_var.get() != 'priority':
            messagebox.showinfo("Observation Planner", "Show the priority list first (source 'Priority list')."); return
        P.remove_priority(names)
        status.set(f"{len(names)} target(s) removed from the priority list.")
        find()

    def show_cluster_vars():
        rs = [r for r in selected_rows() if r['target'].get('type') == 'cluster']
        if not rs:
            messagebox.showinfo("Observation Planner", "Select a cluster (source 'Open clusters')."); return
        t = rs[0]['target']
        pop = tk.Toplevel(root); pop.title(f"Variables in/near {t['name']} ({t['_alias']})"); pop.geometry("760x380")
        tv = ttk.Treeview(pop, columns=("name", "type", "mag", "period", "dist"), show="headings")
        for c, h_, w in (("name", "Name", 150), ("type", "Type", 90), ("mag", "Mag", 100), ("period", "Period (d)", 100),
                         ("dist", "From centre (')", 110)):
            tv.heading(c, text=h_); tv.column(c, width=w)
        for v in t['_vars']:
            p = P._f(v.get('period'))
            tv.insert("", tk.END, values=(v.get('name'), v.get('type') or '', mag_text(v), "" if p is None else f"{p:.5g}",
                                          f"{v['dist_arcmin']:.1f}"))
        tv.pack(fill="both", expand=True, padx=8, pady=8)
        tk.Label(pop, text=f"From the local GCVS list, within 1.5 x the cluster radius ({t['_radius']:g}') or half the "
                           "field of the setup. Use 'Add to Field Variables' in the planner to measure them all.",
                 fg="#555", wraplength=720, justify="left").pack(anchor="w", padx=8, pady=(0, 8))

    def to_lc():
        ts = sel_targets(expand_clusters=False)
        if not ts:
            messagebox.showinfo("Observation Planner", "Select a target first."); return
        hooks['to_light_curves'](ts[0])
        status.set(f"{ts[0].get('name')} set as Light Curves target.")

    def to_fv():
        ts = sel_targets()
        if not ts:
            messagebox.showinfo("Observation Planner", "Select targets first (a selected cluster adds its known variables - "
                                "this one has none in the local GCVS list)." if selected_rows() else
                                "Select targets (or a cluster) first."); return
        n = hooks['to_field_variables'](ts)
        status.set(f"{n} target(s) added to the Field Variables list.")

    def export_csv():
        if not state['rows']:
            return
        path = filedialog.asksaveasfilename(title="Save target list", defaultextension=".csv",
                                            initialfile=f"targets_{date_var.get().strip()}.csv", filetypes=(("CSV", "*.csv"),))
        if not path:
            return
        import csv
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['name', 'ra', 'dec', 'type', 'period', 'max', 'min', 'first', 'last', 'best', 'best_alt',
                        'best_az', 'hours', 'moon_sep', 'note'])
            for r in state['rows']:
                t = r['target']
                w.writerow([t.get('name'), f"{float(t['ra']):.6f}", f"{float(t['dec']):.6f}", t.get('type') or '',
                            t.get('period') or '', t.get('max') or '', t.get('min') or '',
                            r['first'].isoformat(timespec='minutes') if r.get('first') and r.get('hours') else '',
                            r['last'].isoformat(timespec='minutes') if r.get('last') and r.get('hours') else '',
                            r['best'].isoformat(timespec='minutes') if r.get('best') else '',
                            f"{r['best_alt']:.1f}" if r.get('best_alt') is not None else '',
                            f"{r['best_az']:.1f}" if r.get('best_az') is not None else '',
                            f"{r.get('hours', 0):.2f}", f"{r['moon_sep']:.1f}" if r.get('moon_sep') is not None else '',
                            r.get('note', '')])
        status.set(f"List saved: {path}")

    # ---------------------------------------------------------------- v4.38: night plan / NINA export
    def nina_dialog():
        from python import nina_export as NE
        rows = [r for r in selected_rows() if r.get('hours', 0) > 0 and r.get('ok') is not None]
        if not rows:
            rows = [r for r in state['rows'] if r.get('hours', 0) > 0 and r.get('ok') is not None]
            if not rows or not messagebox.askyesno(
                    "Night Plan", f"No targets selected - plan all {len(rows)} observable targets of the list?"):
                return
        setup = _inst.load(setup_var.get()) if setup_var.get() else None
        if setup is None:
            messagebox.showerror("Night Plan", "Choose an instrument setup (with filters and filter positions)."); return
        jd, times = state['jd'], state['times']
        flt = setup.get('filters') or {}
        bands = [b for b in ('V', 'L', 'B', 'R', 'G', 'I') if (flt.get(b) or {}).get('nina_position') is not None]
        win = tk.Toplevel(root); win.title(f"Night Plan for NINA - {setup['name']} - {date_var.get()}"); win.geometry("1180x640")
        top = tk.Frame(win); top.pack(fill="x", padx=8, pady=6)
        fl_var = tk.StringVar(value=(bands[0] if bands else "V"))
        ov_var = tk.StringVar(value="3"); pf_var = tk.StringVar(value="4"); sn_var = tk.StringVar(value="5")
        sky_var = tk.StringVar(value=str((setup.get('planning') or {}).get('sky_mag') or 20.0))
        slew_v, cen_v, gd_v, dith_v = tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        for lab, var, w in (("Filters (e.g. V or V,B or L):", fl_var, 10), ("Overhead per target (min):", ov_var, 4),
                            ("per exposure (s):", pf_var, 4), ("Snapshot exposures:", sn_var, 4),
                            ("Sky (mag/arcsec2):", sky_var, 5)):
            tk.Label(top, text=lab).pack(side=tk.LEFT, padx=(6, 2))
            ttk.Entry(top, textvariable=var, width=w).pack(side=tk.LEFT)
        top2 = tk.Frame(win); top2.pack(fill="x", padx=8)
        for txt, var in (("Slew", slew_v), ("Center (plate solve)", cen_v), ("Start guiding", gd_v), ("Dither", dith_v)):
            ttk.Checkbutton(top2, text=txt, variable=var).pack(side=tk.LEFT, padx=6)
        info = tk.StringVar(value=("Filter positions of the setup: " + ", ".join(
            f"{b}={flt[b].get('fits_name') or b}@{flt[b]['nina_position']}" for b in bands)) if bands else
            "The setup has no filter-wheel positions - enter them in Manage Setups (or 'Filters from NINA File...').")
        tk.Label(top2, textvariable=info, fg="#1a3a5f", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=12)
        cols_ = ("use", "name", "kind", "start", "end", "filters", "exp", "frames", "snr", "why")
        hd = {"use": "Use", "name": "Target", "kind": "What", "start": "Start", "end": "End", "filters": "Filters",
              "exp": "Exp (s)", "frames": "Frames/filter", "snr": "S/N max-min", "why": "Reason / notes"}
        wd = {"use": 40, "name": 120, "kind": 70, "start": 55, "end": 55, "filters": 70, "exp": 60, "frames": 85,
              "snr": 85, "why": 470}
        tv = ttk.Treeview(win, columns=cols_, show="headings", height=14)
        for c in cols_:
            tv.heading(c, text=hd[c]); tv.column(c, width=wd[c], anchor="w" if c in ("name", "why") else "center")
        tv.pack(fill="both", expand=True, padx=8, pady=4)
        st = {'blocks': [], 'filters': [], 'use': {}}
        msg = tk.StringVar(value="")
        tk.Label(win, textvariable=msg, fg="#1a3a5f", wraplength=1150, justify="left").pack(anchor="w", padx=8)

        def local(jdv):
            t0 = times[0]
            return t0 + _dt.timedelta(days=float(jdv - jd[0]))

        def resolve_filters():
            out = []
            for tok in [x.strip() for x in fl_var.get().replace(';', ',').split(',') if x.strip()]:
                b = tok.upper() if tok.upper() in flt else None
                if b is None:
                    b = next((k for k, v in flt.items() if (v.get('fits_name') or '').lower() == tok.lower()), None)
                if b is None or flt[b].get('nina_position') is None:
                    raise ValueError(f"Filter '{tok}': no filter-wheel position in the setup '{setup['name']}'.")
                out.append((flt[b].get('fits_name') or b, int(flt[b]['nina_position']), b))
            if not out:
                raise ValueError("Enter at least one filter.")
            return out

        def compute():
            try:
                fl = resolve_filters()
                ov, pf, sn = float(ov_var.get()), float(pf_var.get()), int(float(sn_var.get()))
                sky = float(sky_var.get())
            except ValueError as e:
                messagebox.showerror("Night Plan", str(e), parent=win); return
            st['filters'] = fl
            items = []
            for r in rows:
                t = r['target']
                band = fl[0][2]
                exp, inf = NE.choose_exposure(t, setup, band, sky_mag=sky)
                k_best = int(np.argmin([abs((tt - r['best']).total_seconds()) for tt in times]))
                items.append({'target': t, 'ok': r['ok'], 'best_jd': jd[k_best], 'exposure': exp, 'info': inf,
                              'filters': [f[0] for f in fl], 'request': NE.block_request(t, jd[0], jd[-1], sn)})
            blocks, rej = NE.make_plan(items, jd, overhead_min=ov, per_frame_s=pf)
            st['blocks'] = blocks
            st['use'] = {id(b): True for b in blocks}
            fill()
            tot = sum(b['end_jd'] - b['start_jd'] for b in blocks) * 24
            gaps = [(blocks[i]['start_jd'] - blocks[i - 1]['end_jd']) * 24 for i in range(1, len(blocks))]
            gap_txt = "; ".join(f"{local(blocks[i - 1]['end_jd']).strftime('%H:%M')}-{local(blocks[i]['start_jd']).strftime('%H:%M')}"
                                for i, g in enumerate(gaps, start=1) if g > 0.05)
            zs = blocks[0]['info'].get('zp_source') if blocks else ''
            msg.set(f"{len(blocks)} target(s) planned, {tot:.1f} h; " +
                    (f"waiting time (written as 'Delay' of the next target): {gap_txt} - add more targets to fill it. "
                     if gap_txt else "") +
                    (f"not placed: {', '.join(x['target'].get('name', '') for x in rej)}. " if rej else "") +
                    f"Exposure times from the zero point ({zs}), saturation limit and sky of the setup - check the "
                    "first frames. Double-click a row to change exposure/frames or to (de)select it.")

        def fill():
            tv.delete(*tv.get_children())
            for b in st['blocks']:
                inf = b['info']
                snr_t = (f"{inf['snr_bright']:.0f}-{inf['snr_faint']:.0f}" if inf.get('snr_faint') is not None else "")
                why = b['request'].get('why', '')
                if b.get('shortened'):
                    why += f" (shortened to {b['shortened'] * 100:.0f} %)"
                if inf.get('note'):
                    why += " | " + inf['note']
                tv.insert("", tk.END, iid=str(id(b)), values=(
                    "yes" if st['use'][id(b)] else "no", b['target'].get('name', ''), b['request']['kind'],
                    local(b['start_jd']).strftime("%H:%M"), local(b['end_jd']).strftime("%H:%M"),
                    ",".join(f[0] for f in st['filters']), f"{b['exposure']:g}", b['frames'], snr_t, why))

        def edit_row(event=None):
            sel = tv.selection()
            if not sel:
                return
            b = next(x for x in st['blocks'] if str(id(x)) == sel[0])
            d = tk.Toplevel(win); d.title(b['target'].get('name', '')); d.transient(win)
            ev, fv, uv = tk.StringVar(value=f"{b['exposure']:g}"), tk.StringVar(value=str(b['frames'])), tk.BooleanVar(value=st['use'][id(b)])
            tk.Label(d, text="Exposure (s):").grid(row=0, column=0, padx=6, pady=4)
            ttk.Entry(d, textvariable=ev, width=8).grid(row=0, column=1)
            tk.Label(d, text="Frames per filter:").grid(row=1, column=0, padx=6, pady=4)
            ttk.Entry(d, textvariable=fv, width=8).grid(row=1, column=1)
            ttk.Checkbutton(d, text="include in the NINA file", variable=uv).grid(row=2, column=0, columnspan=2)

            def ok():
                try:
                    b['exposure'] = float(ev.get()); b['frames'] = max(1, int(float(fv.get())))
                except ValueError:
                    return
                st['use'][id(b)] = bool(uv.get()); d.destroy(); fill()
            ttk.Button(d, text="OK", command=ok).grid(row=3, column=0, columnspan=2, pady=6)
        tv.bind("<Double-1>", edit_row)

        def write():
            use = [b for b in st['blocks'] if st['use'].get(id(b))]
            if not use:
                messagebox.showinfo("Night Plan", "Nothing to write - compute the plan first.", parent=win); return
            folder = NE.default_folder(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            os.makedirs(folder, exist_ok=True)
            safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in setup['name'])
            path = filedialog.asksaveasfilename(parent=win, title="Save NINA target set", initialdir=folder,
                                                initialfile=f"Plan_{date_var.get().strip()}_{safe}.ninaTargetSet.xml",
                                                defaultextension=".xml", filetypes=(("NINA target set", "*.xml"),))
            if not path:
                return
            entries = NE.plan_to_entries(use, setup, [(f[0], f[1]) for f in st['filters']], mode="ROTATE",
                                         start_jd=use[0]['start_jd'], slew=slew_v.get(), center=cen_v.get(),
                                         guide=gd_v.get(), dither=dith_v.get())
            NE.write_target_set(path, entries)
            import csv
            csv_path = re.sub(r'(\.ninaTargetSet)?\.xml$', '', path, flags=re.IGNORECASE) + "_plan.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['target', 'what', 'start_local', 'end_local', 'filters', 'exposure_s', 'frames_per_filter',
                            'snr_max', 'snr_min', 'reason'])
                for b in use:
                    inf = b['info']
                    w.writerow([b['target'].get('name', ''), b['request']['kind'], local(b['start_jd']).isoformat(timespec='minutes'),
                                local(b['end_jd']).isoformat(timespec='minutes'), "+".join(f[0] for f in st['filters']),
                                f"{b['exposure']:g}", b['frames'],
                                "" if inf.get('snr_bright') is None else f"{inf['snr_bright']:.0f}",
                                "" if inf.get('snr_faint') is None else f"{inf['snr_faint']:.0f}",
                                b['request'].get('why', '') + (" | " + inf['note'] if inf.get('note') else "")])
            msg.set(f"Written: {path} ({len(use)} targets) and the schedule {os.path.basename(csv_path)}. In NINA: "
                    f"Sequence (simple sequencer) -> Load Target Set; start it at {local(use[0]['start_jd']).strftime('%H:%M')}.")
        bb = tk.Frame(win); bb.pack(fill="x", padx=8, pady=6)
        ttk.Button(bb, text="Compute Plan", command=compute).pack(side=tk.LEFT, padx=3)
        ttk.Button(bb, text="Write NINA Target Set...", command=write).pack(side=tk.LEFT, padx=3)
        ttk.Button(bb, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=3)
        if bands:
            compute()
        state['nina_dialog'] = {'win': win, 'tree': tv, 'st': st, 'compute': compute, 'write': write, 'msg': msg,
                                'filters_var': fl_var}

    for txt, cmd in (("Add to Priority List", to_priority), ("Remove from Priority List", from_priority),
                     ("Variables in Cluster...", show_cluster_vars), ("Set as Light Curves Target", to_lc),
                     ("Add to Field Variables", to_fv), ("Save List (CSV)...", export_csv),
                     ("Night Plan for NINA...", nina_dialog)):
        ttk.Button(bf if txt not in ("Set as Light Curves Target", "Add to Field Variables", "Save List (CSV)...",
                                     "Night Plan for NINA...") else bf2, text=txt, command=cmd).pack(side=tk.LEFT, padx=3, pady=1)

    setups_refresh()
    return {'find': find, 'refresh': setups_refresh, 'state': state, 'tree': tree, 'nina': nina_dialog}
