package com.ibosoninnov.unitear;

import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import b.b.c.g;
import b.b.c.h;
import c.e.b.hf.e;
import c.e.b.p000if.d;
import c.e.b.p000if.o;
import com.google.android.material.tabs.TabLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/* loaded from: classes2.dex */
public class HistoryActivity extends h {
    public RecyclerView t;
    public boolean u;
    public ImageView v;
    public TabLayout w;
    public int x;
    public d y;
    public ArrayList<String> r = new ArrayList<>(Arrays.asList("Favorites", "History"));
    public HashMap<String, Integer> s = new HashMap<>();
    public final List<e> z = new ArrayList();
    public List<e> A = new ArrayList();

    /* loaded from: classes2.dex */
    public class a implements TabLayout.OnTabSelectedListener {
        public a() {
        }

        @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
        public void onTabReselected(TabLayout.Tab tab) {
        }

        @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
        public void onTabSelected(TabLayout.Tab tab) {
            c.e.b.ef.d dVar;
            new c.e.b.ef.d(HistoryActivity.this.z);
            HistoryActivity.this.A = new ArrayList();
            int i = 0;
            if (tab.getText().equals("Images")) {
                while (i < HistoryActivity.this.z.size()) {
                    if (HistoryActivity.this.z.get(i).image == R.drawable.image) {
                        HistoryActivity historyActivity = HistoryActivity.this;
                        historyActivity.A.add(historyActivity.z.get(i));
                    }
                    i++;
                }
                dVar = new c.e.b.ef.d(HistoryActivity.this.A);
            } else if (tab.getText().equals("3D")) {
                while (i < HistoryActivity.this.z.size()) {
                    if (HistoryActivity.this.z.get(i).image == R.drawable.d_model_icon) {
                        HistoryActivity historyActivity2 = HistoryActivity.this;
                        historyActivity2.A.add(historyActivity2.z.get(i));
                    }
                    i++;
                }
                dVar = new c.e.b.ef.d(HistoryActivity.this.A);
            } else if (tab.getText().equals("360°")) {
                while (i < HistoryActivity.this.z.size()) {
                    if (HistoryActivity.this.z.get(i).image == R.drawable.degree) {
                        HistoryActivity historyActivity3 = HistoryActivity.this;
                        historyActivity3.A.add(historyActivity3.z.get(i));
                    }
                    i++;
                }
                dVar = new c.e.b.ef.d(HistoryActivity.this.A);
            } else if (tab.getText().equals("Video")) {
                while (i < HistoryActivity.this.z.size()) {
                    if (HistoryActivity.this.z.get(i).image == R.drawable.video_icon) {
                        HistoryActivity historyActivity4 = HistoryActivity.this;
                        historyActivity4.A.add(historyActivity4.z.get(i));
                    }
                    i++;
                }
                dVar = new c.e.b.ef.d(HistoryActivity.this.A);
            } else if (tab.getText().equals("Slideshow")) {
                while (i < HistoryActivity.this.z.size()) {
                    if (HistoryActivity.this.z.get(i).image == R.drawable.image_slideshow) {
                        HistoryActivity historyActivity5 = HistoryActivity.this;
                        historyActivity5.A.add(historyActivity5.z.get(i));
                    }
                    i++;
                }
                dVar = new c.e.b.ef.d(HistoryActivity.this.A);
            } else if (tab.getText().equals("Podcast")) {
                while (i < HistoryActivity.this.z.size()) {
                    if (HistoryActivity.this.z.get(i).image == R.drawable.podcast_icon) {
                        HistoryActivity historyActivity6 = HistoryActivity.this;
                        historyActivity6.A.add(historyActivity6.z.get(i));
                    }
                    i++;
                }
                dVar = new c.e.b.ef.d(HistoryActivity.this.A);
            } else {
                while (i < HistoryActivity.this.z.size()) {
                    HistoryActivity historyActivity7 = HistoryActivity.this;
                    historyActivity7.A.add(historyActivity7.z.get(i));
                    i++;
                }
                dVar = new c.e.b.ef.d(HistoryActivity.this.A);
            }
            HistoryActivity.this.t.setAdapter(dVar);
        }

        @Override // com.google.android.material.tabs.TabLayout.BaseOnTabSelectedListener
        public void onTabUnselected(TabLayout.Tab tab) {
        }
    }

    /* loaded from: classes2.dex */
    public class b implements c.e.b.p000if.h {
        public b() {
        }

        @Override // c.e.b.p000if.h
        public void a(View view, int i) {
            String str;
            HistoryActivity historyActivity = HistoryActivity.this;
            if (historyActivity.u) {
                historyActivity.z.get(i).isCheck.c(!HistoryActivity.this.z.get(i).isCheck.f2335c);
                int i2 = 0;
                for (int i3 = 0; i3 < HistoryActivity.this.z.size(); i3++) {
                    if (HistoryActivity.this.z.get(i3).isCheck.f2335c) {
                        i2++;
                    }
                }
                if (i2 == 0) {
                    HistoryActivity.this.v.setVisibility(8);
                    HistoryActivity.this.u = false;
                    return;
                }
                return;
            }
            e eVar = historyActivity.A.get(i);
            if (eVar != null) {
                str = eVar.id;
                Log.d("History", "Pos = " + i + " " + str);
            } else {
                str = "";
            }
            Log.d("jsonData", "jsonData:  " + str);
            d dVar = HistoryActivity.this.y;
            dVar.f4872b.putString("fromHistory", str);
            dVar.f4872b.apply();
            HistoryActivity.this.finish();
        }

        @Override // c.e.b.p000if.h
        public void b(View view, int i) {
            HistoryActivity historyActivity = HistoryActivity.this;
            if (historyActivity.x == 1 || historyActivity.u) {
                return;
            }
            historyActivity.v.setVisibility(0);
            HistoryActivity historyActivity2 = HistoryActivity.this;
            historyActivity2.u = true;
            historyActivity2.z.get(i).isCheck.c(!HistoryActivity.this.z.get(i).isCheck.f2335c);
        }
    }

    /* loaded from: classes2.dex */
    public class c implements DialogInterface.OnClickListener {
        public c() {
        }

        @Override // android.content.DialogInterface.OnClickListener
        public void onClick(DialogInterface dialogInterface, int i) {
            HistoryActivity.this.v.setVisibility(8);
            HistoryActivity historyActivity = HistoryActivity.this;
            historyActivity.u = false;
            if (historyActivity.x == 1) {
                historyActivity.z.clear();
                HistoryActivity.this.t.setAdapter(null);
                d dVar = HistoryActivity.this.y;
                dVar.f4872b.putString("history", "");
                dVar.f4872b.apply();
                return;
            }
            d dVar2 = historyActivity.y;
            dVar2.f4872b.putString("fav", "");
            dVar2.f4872b.apply();
            HistoryActivity.this.z.clear();
            HistoryActivity.this.t.setAdapter(null);
            HistoryActivity.this.t.setAdapter(new c.e.b.ef.d(HistoryActivity.this.z));
        }
    }

    @Override // b.b.c.h, android.app.Activity, android.view.ContextThemeWrapper, android.content.ContextWrapper
    public void attachBaseContext(Context context) {
        super.attachBaseContext(context);
    }

    public void onBack(View view) {
        finish();
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_history);
        this.x = getIntent().getExtras().getInt("position");
        this.v = (ImageView) findViewById(R.id.imDelete);
        TabLayout tabLayout = (TabLayout) findViewById(R.id.tabs);
        this.w = tabLayout;
        tabLayout.addTab(tabLayout.newTab().setText("All"));
        this.w.addOnTabSelectedListener((TabLayout.OnTabSelectedListener) new a());
        HashMap<String, Integer> hashMap = this.s;
        Integer valueOf = Integer.valueOf((int) R.drawable.video_icon);
        hashMap.put("1", valueOf);
        HashMap<String, Integer> hashMap2 = this.s;
        Integer valueOf2 = Integer.valueOf((int) R.drawable.d_model_icon);
        hashMap2.put("2", valueOf2);
        this.s.put("3", valueOf2);
        HashMap<String, Integer> hashMap3 = this.s;
        Integer valueOf3 = Integer.valueOf((int) R.drawable.image);
        hashMap3.put("4", valueOf3);
        this.s.put("5", valueOf2);
        this.s.put("6", Integer.valueOf((int) R.drawable.image_slideshow));
        this.s.put("7", Integer.valueOf((int) R.drawable.podcast_icon));
        this.s.put("8", Integer.valueOf((int) R.drawable.degree));
        this.s.put("9", valueOf3);
        this.s.put("10", valueOf);
        this.s.put("11", valueOf);
        this.s.put("12", valueOf);
        this.s.put("13", valueOf);
        this.s.put("14", valueOf);
        this.s.put("15", valueOf);
        this.s.put("16", valueOf);
        this.s.put("17", valueOf);
        ((TextView) findViewById(R.id.tvHeading)).setText(this.r.get(this.x));
        this.t = (RecyclerView) findViewById(R.id.recyclerHistory);
        this.y = new d(this);
        this.t.setLayoutManager(new GridLayoutManager(this, 2));
        String string = this.y.f4871a.getString("history", "");
        String string2 = this.y.f4871a.getString("fav", "");
        if (this.x == 1) {
            Log.d("jsonData", "jsonData:  " + string);
            r2 = string.isEmpty() ? null : string.split(",");
            this.v.setVisibility(0);
            this.v.setImageResource(R.drawable.history_clear_icon);
        } else if (!string2.isEmpty()) {
            r2 = string.split(",");
        }
        if (r2 != null) {
            for (String str : r2) {
                e eVar = new e();
                eVar.image = this.s.get("2").intValue();
                eVar.name = "";
                eVar.id = str;
                this.z.add(eVar);
                this.A.add(eVar);
            }
            this.t.setAdapter(new c.e.b.ef.d(this.z));
            RecyclerView recyclerView = this.t;
            recyclerView.addOnItemTouchListener(new o(this, recyclerView, new b()));
        }
    }

    public void onDelete(View view) {
        new g.a(this).setIcon(17301543).setTitle("Delete").setMessage(this.x == 1 ? "Are you sure you want to clear history?" : "Are you sure you want to delete these items?").setPositiveButton("Yes", new c()).setNegativeButton("No", (DialogInterface.OnClickListener) null).show();
    }
}