package com.ibosoninnov.unitear;

import android.graphics.Rect;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.util.DisplayMetrics;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.appcompat.widget.SearchView;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import b.b.c.h;
import c.e.b.ef.a;
import c.e.b.p000if.d;
import c.e.b.p000if.n;
import c.e.b.p000if.o;
import c.e.b.vb;
import c.e.b.wb;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/* loaded from: classes2.dex */
public class ARGallerySubActivity extends h implements SearchView.l {
    public String A;
    public String B;
    public d C;
    public RecyclerView r;
    public List<c.e.b.hf.b> s = new ArrayList();
    public List<c.e.b.hf.a> t;
    public c.e.b.ef.a u;
    public RelativeLayout v;
    public ImageView w;
    public ImageView x;
    public EditText y;
    public TextView z;

    /* loaded from: classes2.dex */
    public class a implements View.OnClickListener {
        public a() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            ARGallerySubActivity.this.getWindow().setWindowAnimations(0);
            ARGallerySubActivity.this.overridePendingTransition(0, 0);
            ARGallerySubActivity.this.finish();
        }
    }

    /* loaded from: classes2.dex */
    public class b implements TextWatcher {
        public b() {
        }

        @Override // android.text.TextWatcher
        public void afterTextChanged(Editable editable) {
            c.e.b.ef.a aVar = ARGallerySubActivity.this.u;
            Objects.requireNonNull(aVar);
            new a.C0088a().filter(ARGallerySubActivity.this.y.getText().toString());
        }

        @Override // android.text.TextWatcher
        public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
        }

        @Override // android.text.TextWatcher
        public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            if (TextUtils.isEmpty(ARGallerySubActivity.this.y.getText())) {
                ARGallerySubActivity.this.x.setVisibility(8);
            } else {
                ARGallerySubActivity.this.x.setVisibility(0);
            }
        }
    }

    /* loaded from: classes2.dex */
    public class c implements View.OnClickListener {
        public c() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            ARGallerySubActivity.this.y.getText().clear();
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
        c.e.b.ef.a aVar = this.u;
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
        setContentView(R.layout.activity_argallery_sub);
        this.C = new d(this);
        Bundle extras = getIntent().getExtras();
        if (extras != null) {
            if (extras.containsKey("title")) {
                this.A = extras.getString("title");
            }
            if (extras.containsKey("jsonData")) {
                this.B = extras.getString("jsonData");
            }
        }
        this.r = (RecyclerView) findViewById(R.id.listOfAR);
        DisplayMetrics displayMetrics = getApplicationContext().getResources().getDisplayMetrics();
        int i = ((float) displayMetrics.widthPixels) / displayMetrics.density > 480.0f ? 3 : 2;
        this.v = (RelativeLayout) findViewById(R.id.emptyViewLayout);
        this.r.setLayoutManager(new GridLayoutManager(this, i));
        this.r.addItemDecoration(new n(this, R.dimen.recyclerItemSpacing));
        c.e.b.ef.a aVar = new c.e.b.ef.a(this.s, this, new vb(this));
        this.u = aVar;
        this.r.setAdapter(aVar);
        RecyclerView recyclerView = this.r;
        recyclerView.addOnItemTouchListener(new o(this, recyclerView, new wb(this)));
        String str = this.B;
        this.t = new ArrayList();
        this.t = c.e.b.hf.a.a(str);
        for (int i2 = 0; i2 < this.t.size(); i2++) {
            c.e.b.hf.a aVar2 = this.t.get(i2);
            if (aVar2.category.equals(this.A)) {
                c.e.b.hf.b bVar = new c.e.b.hf.b();
                bVar.imageUrl.add(aVar2.thumbnail_url);
                bVar.name = aVar2.prefab_name;
                bVar.id = aVar2.id;
                bVar.category = aVar2.category;
                bVar.glbFile = aVar2.file_loc;
                this.s.add(bVar);
            }
        }
        this.u.notifyDataSetChanged();
        this.y = (EditText) findViewById(R.id.searchet);
        this.w = (ImageView) findViewById(R.id.backBtn);
        this.x = (ImageView) findViewById(R.id.clearbutton);
        TextView textView = (TextView) findViewById(R.id.title);
        this.z = textView;
        textView.setText(this.A);
        this.w.setOnClickListener(new a());
        this.y.addTextChangedListener(new b());
        this.x.setOnClickListener(new c());
    }

    @Override // android.app.Activity
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.options_menu, menu);
        return true;
    }

    @Override // android.app.Activity
    public boolean onOptionsItemSelected(MenuItem menuItem) {
        if (menuItem.getItemId() != 16908332) {
            return super.onOptionsItemSelected(menuItem);
        }
        finish();
        return true;
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public void onWindowFocusChanged(boolean z) {
        super.onWindowFocusChanged(z);
        if (z) {
            getWindow().getDecorView().setSystemUiVisibility(5894);
        }
    }
}