"""
GUI of the observation planner (v4.37) - built into a frame of the main window.

build(parent, root, vars_dict, hooks) with hooks:
  'to_light_curves'(target)   - set the Light Curves target
  'to_field_variables'(list)  - add targets to the Field Variables list
"""
import datetime as _dt
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

from photometry import planner as P
from photometry import instruments as _inst


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
    tk.Label(f1, textvariable=site_lbl, bg="white", fg="#1a3a5f", font=("Segoe UI", 8)).grid(row=0, column=2, columnspan=6, sticky="w", padx=6)
    tk.Label(f1, text="Night of (date):", bg="white").grid(row=1, column=0, sticky="w", padx=6)
    ttk.Entry(f1, textvariable=date_var, width=12).grid(row=1, column=1, sticky="w")
    tk.Label(f1, text="from", bg="white").grid(row=1, column=2, sticky="e")
    ttk.Entry(f1, textvariable=t0_var, width=7).grid(row=1, column=3, sticky="w", padx=3)
    tk.Label(f1, text="to", bg="white").grid(row=1, column=4, sticky="e")
    ttk.Entry(f1, textvariable=t1_var, width=7).grid(row=1, column=5, sticky="w", padx=3)
    tk.Label(f1, text="(local time HH:MM, or 'dusk' / 'dawn')   Darkness:", bg="white").grid(row=1, column=6, sticky="w")
    ttk.Combobox(f1, textvariable=dark_var, width=15, state="readonly",
                 values=["civil (-6)", "nautical (-12)", "astronomical (-18)", "any (sun ignored)"]).grid(row=1, column=7, sticky="w", padx=3)
    night_lbl = tk.StringVar(value="")
    tk.Label(f1, textvariable=night_lbl, bg="white", fg="#555", font=("Segoe UI", 8), justify="left").grid(
        row=2, column=0, columnspan=8, sticky="w", padx=6, pady=(0, 3))

    f2 = tk.LabelFrame(box, text="Selection", bg="white"); f2.pack(fill="x", pady=3)
    alt_var = V("plan_alt_min", "30"); sector_var = V("plan_sector", "any"); moon_var = V("plan_moon_min", "20")
    mb_var = V("plan_mag_bright", ""); mf_var = V("plan_mag_faint", "")
    types_var = V("plan_types", ""); pmin_var = V("plan_pmin", ""); pmax_var = V("plan_pmax", ""); amp_var = V("plan_amp_min", "")
    minh_var = V("plan_min_hours", "0.5")
    row = 0
    for lab, var, w, col in (("Altitude >= (deg):", alt_var, 5, 0), ("Moon distance >= (deg):", moon_var, 5, 4),
                             ("Observable at least (h):", minh_var, 5, 6)):
        tk.Label(f2, text=lab, bg="white").grid(row=row, column=col, sticky="w", padx=6, pady=2)
        ttk.Entry(f2, textvariable=var, width=w).grid(row=row, column=col + 1, sticky="w")
    tk.Label(f2, text="Azimuth:", bg="white").grid(row=0, column=2, sticky="w", padx=6)
    ttk.Combobox(f2, textvariable=sector_var, width=12, values=list(P.SECTORS.keys()) + ["60-200"]).grid(row=0, column=3, sticky="w")
    row = 1
    tk.Label(f2, text="Magnitude from", bg="white").grid(row=row, column=0, sticky="w", padx=6)
    ttk.Entry(f2, textvariable=mb_var, width=5).grid(row=row, column=1, sticky="w")
    tk.Label(f2, text="to", bg="white").grid(row=row, column=2, sticky="e")
    ttk.Entry(f2, textvariable=mf_var, width=5).grid(row=row, column=3, sticky="w")
    tk.Label(f2, text="(empty = limits of the setup)", bg="white", fg="#777", font=("Segoe UI", 8)).grid(row=row, column=4, columnspan=2, sticky="w")
    row = 2
    tk.Label(f2, text="Types:", bg="white").grid(row=row, column=0, sticky="w", padx=6)
    ttk.Entry(f2, textvariable=types_var, width=18).grid(row=row, column=1, columnspan=2, sticky="w")
    tk.Label(f2, text="e.g. EA EW DSCT* M (empty = all)", bg="white", fg="#777", font=("Segoe UI", 8)).grid(row=row, column=3, columnspan=2, sticky="w")
    tk.Label(f2, text="Period (d) from", bg="white").grid(row=row, column=5, sticky="e")
    ttk.Entry(f2, textvariable=pmin_var, width=6).grid(row=row, column=6, sticky="w")
    tk.Label(f2, text="to", bg="white").grid(row=row, column=7, sticky="w")
    ttk.Entry(f2, textvariable=pmax_var, width=6).grid(row=row, column=8, sticky="w")
    tk.Label(f2, text="Amplitude >=", bg="white").grid(row=row, column=9, sticky="e", padx=(6, 0))
    ttk.Entry(f2, textvariable=amp_var, width=5).grid(row=row, column=10, sticky="w")

    f3 = tk.Frame(box, bg="white"); f3.pack(fill="x", pady=4)
    source_var = V("plan_source", "gcvs")
    for txt, val in (("Local GCVS list", "gcvs"), ("Priority list", "priority"), ("Open clusters", "clusters")):
        ttk.Radiobutton(f3, text=txt, variable=source_var, value=val).pack(side=tk.LEFT, padx=6)
    find_btn = tk.Button(f3, text="Find Targets", bg="#388e3c", fg="white", font=("Arial", 10, "bold"), padx=14)
    find_btn.pack(side=tk.LEFT, padx=12)
    status = tk.StringVar(value="")
    tk.Label(f3, textvariable=status, bg="white", fg="#1a3a5f", font=("Segoe UI", 9), wraplength=520, justify="left").pack(side=tk.LEFT)

    # ---------------------------------------------------------------- results
    cols = ("name", "type", "mag", "period", "window", "best", "alt", "hours", "moon", "note")
    heads = {"name": "Name", "type": "Type", "mag": "Mag (max-min)", "period": "Period (d)",
             "window": "Observable (local)", "best": "Best", "alt": "Alt / Az at best", "hours": "Hours",
             "moon": "Moon dist", "note": "Note"}
    widths = {"name": 130, "type": 70, "mag": 95, "period": 75, "window": 110, "best": 55, "alt": 85, "hours": 50,
              "moon": 65, "note": 230}
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

    bf = tk.Frame(box, bg="white"); bf.pack(fill="x", pady=4)
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
            return {'name': str(t.get('name', '')).lower(), 'type': str(t.get('type') or ''),
                    'mag': P._f(t.get('max')) or 99, 'period': P._f(t.get('period')) or 1e9,
                    'window': r['first'].timestamp() if r.get('first') else 0, 'best': r['best'].timestamp() if r.get('best') else 0,
                    'alt': r.get('best_alt', -99), 'hours': r.get('hours', 0), 'moon': r.get('moon_sep', 0),
                    'note': r.get('note', '')}[key]
        for r in sorted(state['rows'], key=sk, reverse=rev):
            t = r['target']
            p = P._f(t.get('period'))
            vals = (t.get('name', ''), t.get('type') or '', mag_text(t), "" if p is None else f"{p:.5g}",
                    f"{fmt_t(r.get('first'))}-{fmt_t(r.get('last'))}" if r.get('hours') else "",
                    fmt_t(r.get('best')) if r.get('hours') else "",
                    f"{r['best_alt']:.0f} / {P.compass(r['best_az'])}" if r.get('best_az') is not None else "",
                    f"{r.get('hours', 0):.1f}", f"{r['moon_sep']:.0f}" if r.get('moon_sep') is not None else "",
                    r.get('note', ''))
            iid = tree.insert("", tk.END, values=vals)
            state['items'][iid] = r

    def sort_by(c):
        k, rev = state['sort']
        state['sort'] = (c, not rev if k == c else c in ('hours', 'alt', 'moon'))
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
        ticks = np.arange(0, x[-1] + 0.01, 1.0 if x[-1] <= 12 else 2.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels([(times[0] + _dt.timedelta(hours=float(h))).strftime("%H:%M") for h in ticks], fontsize=7)
        ax.set_ylim(0, 90); ax.set_xlim(x[0], x[-1])
        ax.set_ylabel("Altitude (deg)", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_title("Altitude (thick: observable in the chosen window/sector)", fontsize=8)
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
            sector = P.parse_sector(sector_var.get())
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
                                         f"alt {r['best_alt']:.0f} {P.compass(r['best_az'])}")
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
                    note_extra = (f"{len(cl)} clusters (photometry_refstars/open_clusters.csv); variables from the "
                                  "local GCVS list" + ("" if cat else " - no local GCVS list found"))
                state.update(rows=rows, times=times, night=night)

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

    for txt, cmd in (("Add to Priority List", to_priority), ("Remove from Priority List", from_priority),
                     ("Variables in Cluster...", show_cluster_vars), ("Set as Light Curves Target", to_lc),
                     ("Add to Field Variables", to_fv), ("Save List (CSV)...", export_csv)):
        ttk.Button(bf, text=txt, command=cmd).pack(side=tk.LEFT, padx=3)

    setups_refresh()
    return {'find': find, 'refresh': setups_refresh, 'state': state, 'tree': tree}
