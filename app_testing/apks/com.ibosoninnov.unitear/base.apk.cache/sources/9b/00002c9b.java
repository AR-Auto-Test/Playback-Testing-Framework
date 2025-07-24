package com.ibosoninnov.unitear;

import android.content.Context;
import android.graphics.Rect;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import androidx.appcompat.widget.SearchView;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;
import b.b.c.h;
import c.e.b.ac;
import c.e.b.bc;
import c.e.b.c0;
import c.e.b.cc;
import c.e.b.ef.a;
import c.e.b.hf.b;
import c.e.b.p000if.d;
import c.e.b.p000if.n;
import c.e.b.p000if.o;
import c.e.b.tb;
import c.e.b.ub;
import com.ibosoninnov.unitear.ARGalleryActivity;
import f.v;
import f.x;
import f.y;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/* loaded from: classes2.dex */
public class ARGalleryActivity extends h implements SearchView.l {
    public static boolean r = false;
    public static String s;
    public String A;
    public RelativeLayout B;
    public EditText C;
    public SwipeRefreshLayout D;
    public d E;
    public ProgressBar F;
    public Context t;
    public ImageView u;
    public ImageView v;
    public List<b> w = new ArrayList();
    public List<b> x = new ArrayList();
    public List<c.e.b.hf.a> y;
    public c.e.b.ef.a z;

    /* loaded from: classes2.dex */
    public class a implements TextWatcher {
        public a() {
        }

        @Override // android.text.TextWatcher
        public void afterTextChanged(Editable editable) {
            c.e.b.ef.a aVar = ARGalleryActivity.this.z;
            Objects.requireNonNull(aVar);
            new a.C0088a().filter(ARGalleryActivity.this.C.getText().toString());
        }

        @Override // android.text.TextWatcher
        public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        }

        @Override // android.text.TextWatcher
        public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            if (TextUtils.isEmpty(ARGalleryActivity.this.C.getText())) {
                ARGalleryActivity.this.v.setVisibility(8);
            } else {
                ARGalleryActivity.this.v.setVisibility(0);
            }
        }
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public boolean dispatchTouchEvent(MotionEvent motionEvent) {
        if (motionEvent.getAction() == 0) {
            View currentFocus = getCurrentFocus();
            if (currentFocus instanceof EditText) {
                Rect rect = new Rect();
                currentFocus.getGlobalVisibleRect(rect);
                if (!rect.contains((int) motionEvent.getRawX(), (int) motionEvent.getRawY())) {
                    currentFocus.clearFocus();
                    ((InputMethodManager) getSystemService("input_method")).hideSoftInputFromWindow(currentFocus.getWindowToken(), 0);
                }
            }
        }
        return super.dispatchTouchEvent(motionEvent);
    }

    @Override // androidx.appcompat.widget.SearchView.l
    public boolean e(String str) {
        c.e.b.ef.a aVar = this.z;
        Objects.requireNonNull(aVar);
        new a.C0088a().filter(str);
        return false;
    }

    @Override // androidx.appcompat.widget.SearchView.l
    public boolean g(String str) {
        return false;
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getWindow().getDecorView().setSystemUiVisibility(2);
        getWindow().setFlags(1024, 1024);
        getWindow().addFlags(1536);
        getWindow().addFlags(128);
        setContentView(R.layout.activity_argallery);
        this.t = this;
        this.u = (ImageView) findViewById(R.id.backBtn);
        this.v = (ImageView) findViewById(R.id.clearbutton);
        this.E = new d(this);
        r = false;
        this.F = (ProgressBar) findViewById(R.id.progressbar);
        this.C = (EditText) findViewById(R.id.searchet);
        RecyclerView recyclerView = (RecyclerView) findViewById(R.id.listOfAR);
        DisplayMetrics displayMetrics = getApplicationContext().getResources().getDisplayMetrics();
        int i = ((float) displayMetrics.widthPixels) / displayMetrics.density > 480.0f ? 3 : 2;
        this.B = (RelativeLayout) findViewById(R.id.emptyViewLayout);
        recyclerView.setLayoutManager(new GridLayoutManager(this, i));
        recyclerView.addItemDecoration(new n(this, R.dimen.recyclerItemSpacing));
        c.e.b.ef.a aVar = new c.e.b.ef.a(this.w, this.x, this, new a.b() { // from class: c.e.b.b0
            @Override // c.e.b.ef.a.b
            public final void a(final boolean z) {
                final ARGalleryActivity aRGalleryActivity = ARGalleryActivity.this;
                Objects.requireNonNull(aRGalleryActivity);
                Log.d("ARGallery", "Empty " + z);
                aRGalleryActivity.runOnUiThread(new Runnable() { // from class: c.e.b.a0
                    @Override // java.lang.Runnable
                    public final void run() {
                        ARGalleryActivity aRGalleryActivity2 = ARGalleryActivity.this;
                        if (z) {
                            aRGalleryActivity2.B.setVisibility(0);
                        } else {
                            aRGalleryActivity2.B.setVisibility(8);
                        }
                    }
                });
            }
        });
        this.z = aVar;
        recyclerView.setAdapter(aVar);
        recyclerView.addOnItemTouchListener(new o(this, recyclerView, new tb(this)));
        SwipeRefreshLayout swipeRefreshLayout = (SwipeRefreshLayout) findViewById(R.id.swiperefresh);
        this.D = swipeRefreshLayout;
        swipeRefreshLayout.setColorSchemeResources(17170451, 17170452, 17170456, 17170454);
        this.D.setOnRefreshListener(new c0(this));
        w();
        this.C.addTextChangedListener(new a());
        this.u.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.e0
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ARGalleryActivity.this.finish();
            }
        });
        this.v.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.f0
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                ARGalleryActivity.this.C.getText().clear();
            }
        });
    }

    @Override // android.app.Activity
    public boolean onCreateOptionsMenu(Menu menu) {
        return true;
    }

    @Override // android.app.Activity
    public boolean onOptionsItemSelected(MenuItem menuItem) {
        if (menuItem.getItemId() == 16908332) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(menuItem);
    }

    @Override // b.q.b.d, android.app.Activity
    public void onResume() {
        super.onResume();
        d dVar = this.E;
        if (dVar != null && !dVar.f4871a.getString("arGalleryFile", "").isEmpty()) {
            finish();
        }
        if (r) {
            finish();
        }
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public void onWindowFocusChanged(boolean z) {
        super.onWindowFocusChanged(z);
        if (z) {
            getWindow().getDecorView().setSystemUiVisibility(5894);
        }
    }

    public final void v(String str) {
        boolean z;
        this.y = new ArrayList();
        this.y = c.e.b.hf.a.a(str);
        for (int i = 0; i < this.y.size(); i++) {
            c.e.b.hf.a aVar = this.y.get(i);
            b bVar = new b();
            bVar.imageUrl.add(aVar.thumbnail_url);
            bVar.name = aVar.prefab_name;
            bVar.id = aVar.id;
            bVar.category = aVar.category;
            bVar.glbFile = aVar.file_loc;
            if (!this.w.isEmpty()) {
                for (b bVar2 : this.w) {
                    if (aVar.category.equals(bVar2.category)) {
                        bVar2.imageUrl.add(aVar.thumbnail_url);
                        z = false;
                        break;
                    }
                }
            }
            z = true;
            if (z) {
                this.w.add(bVar);
            }
            this.x.add(bVar);
        }
        this.z.notifyDataSetChanged();
    }

    public final void w() {
        y a2;
        final String str;
        boolean z = false;
        if (new File(getFilesDir().getAbsolutePath() + "/storage.json").exists() && this.E.f4871a.getBoolean("gallery_updated", false)) {
            try {
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(openFileInput("storage.json")));
                StringBuilder sb = new StringBuilder();
                while (true) {
                    String readLine = bufferedReader.readLine();
                    if (readLine == null) {
                        break;
                    }
                    sb.append(readLine);
                }
                str = sb.toString();
            } catch (IOException unused) {
                str = null;
            }
            s = str;
            runOnUiThread(new Runnable() { // from class: c.e.b.d0
                @Override // java.lang.Runnable
                public final void run() {
                    ARGalleryActivity aRGalleryActivity = ARGalleryActivity.this;
                    String str2 = str;
                    aRGalleryActivity.A = str2;
                    aRGalleryActivity.v(str2);
                    aRGalleryActivity.D.setRefreshing(false);
                    aRGalleryActivity.F.setVisibility(8);
                }
            });
            return;
        }
        try {
            FileOutputStream openFileOutput = openFileOutput("storage.json", 0);
            openFileOutput.write("{}".getBytes());
            openFileOutput.close();
            z = true;
        } catch (IOException unused2) {
        }
        if (z) {
            cc ccVar = new cc(new ub(this));
            v vVar = cc.f4613a;
            if (vVar != null) {
                vVar.f6122d.a();
            }
            v.b bVar = new v.b();
            TimeUnit timeUnit = TimeUnit.SECONDS;
            bVar.a(10L, timeUnit);
            bVar.c(10L, timeUnit);
            bVar.b(15L, timeUnit);
            cc.f4613a = new v(bVar);
            if (ac.f4547a.f4552f) {
                y.a aVar = new y.a();
                aVar.d("https://www.unitear.com/unitear/ground_plane_new");
                a2 = aVar.a();
            } else {
                y.a aVar2 = new y.a();
                aVar2.d("https://www.unitear.com/unitear/ground_plane_new");
                a2 = aVar2.a();
            }
            ((x) cc.f4613a.a(a2)).b(new bc(ccVar));
        }
    }
}