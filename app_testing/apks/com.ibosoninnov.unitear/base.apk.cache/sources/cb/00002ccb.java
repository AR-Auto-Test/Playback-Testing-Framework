package com.ibosoninnov.unitear;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.RelativeLayout;
import android.widget.Switch;
import b.b.c.h;
import b.b.c.u;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;

/* loaded from: classes2.dex */
public class SettingsActivity extends h {
    public static final /* synthetic */ int r = 0;
    public boolean A = false;
    public boolean B = false;
    public RelativeLayout s;
    public RelativeLayout t;
    public RelativeLayout u;
    public RelativeLayout v;
    public RelativeLayout w;
    public Switch x;
    public Switch y;
    public c.e.b.p000if.d z;

    /* loaded from: classes2.dex */
    public class a implements View.OnClickListener {
        public a() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            SettingsActivity settingsActivity = SettingsActivity.this;
            int i = AboutTheApp.r;
            Intent intent = new Intent(settingsActivity, AboutTheApp.class);
            intent.putExtra("position", 1);
            settingsActivity.startActivity(intent);
        }
    }

    /* loaded from: classes2.dex */
    public class b implements View.OnClickListener {
        public b() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            SettingsActivity settingsActivity = SettingsActivity.this;
            int i = SettingsActivity.r;
            Objects.requireNonNull(settingsActivity);
            Intent intent = new Intent("android.intent.action.VIEW");
            intent.setData(Uri.parse("https://www.unitear.com/privacy-policy"));
            settingsActivity.startActivity(intent);
        }
    }

    /* loaded from: classes2.dex */
    public class c implements View.OnClickListener {
        public c() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            SettingsActivity settingsActivity = SettingsActivity.this;
            int i = SettingsActivity.r;
            Objects.requireNonNull(settingsActivity);
            Intent intent = new Intent("android.intent.action.VIEW");
            intent.setData(Uri.parse("https://www.unitear.com/terms"));
            settingsActivity.startActivity(intent);
        }
    }

    /* loaded from: classes2.dex */
    public class d implements View.OnClickListener {
        public d() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            SettingsActivity.this.startActivity(new Intent("android.intent.action.VIEW", Uri.parse("https://www.instagram.com/unitear_augmented_reality/")));
        }
    }

    /* loaded from: classes2.dex */
    public class e implements View.OnClickListener {
        public e() {
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            SettingsActivity.this.startActivity(new Intent("android.intent.action.VIEW", Uri.parse("https://www.facebook.com/theUniteAR")));
        }
    }

    /* loaded from: classes2.dex */
    public class f implements CompoundButton.OnCheckedChangeListener {
        public f() {
        }

        @Override // android.widget.CompoundButton.OnCheckedChangeListener
        public void onCheckedChanged(CompoundButton compoundButton, boolean z) {
            c.e.b.p000if.d dVar = SettingsActivity.this.z;
            dVar.f4872b.putBoolean("Audio", z);
            dVar.f4872b.apply();
        }
    }

    /* loaded from: classes2.dex */
    public class g implements CompoundButton.OnCheckedChangeListener {
        public g() {
        }

        @Override // android.widget.CompoundButton.OnCheckedChangeListener
        public void onCheckedChanged(CompoundButton compoundButton, boolean z) {
            c.e.b.p000if.d dVar = SettingsActivity.this.z;
            dVar.f4872b.putBoolean("Gyro", z);
            dVar.f4872b.apply();
        }
    }

    public SettingsActivity() {
        new ArrayList(Arrays.asList("About the app", "Send feedback"));
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
        setContentView(R.layout.activity_settings);
        r().c(true);
        b.b.c.a r2 = r();
        ((u) r2).f620g.setTitle(getResources().getString(R.string.settings));
        c.e.b.p000if.d dVar = new c.e.b.p000if.d(this);
        this.z = dVar;
        this.A = dVar.f4871a.getBoolean("Audio", false);
        this.B = this.z.f4871a.getBoolean("Gyro", false);
        RelativeLayout relativeLayout = (RelativeLayout) findViewById(R.id.aboutAppBtn);
        this.s = relativeLayout;
        relativeLayout.setOnClickListener(new a());
        RelativeLayout relativeLayout2 = (RelativeLayout) findViewById(R.id.privacyBtn);
        this.u = relativeLayout2;
        relativeLayout2.setOnClickListener(new b());
        this.t = (RelativeLayout) findViewById(R.id.termsOfServiceBtn);
        this.w = (RelativeLayout) findViewById(R.id.facebookBtn);
        this.v = (RelativeLayout) findViewById(R.id.instagramBtn);
        this.t.setOnClickListener(new c());
        this.v.setOnClickListener(new d());
        this.w.setOnClickListener(new e());
        Switch r4 = (Switch) findViewById(R.id.audioToggleBtn);
        this.x = r4;
        if (this.A) {
            r4.setChecked(true);
        }
        this.x.setOnCheckedChangeListener(new f());
        Switch r42 = (Switch) findViewById(R.id.gyroToggleBtn);
        this.y = r42;
        if (this.B) {
            r42.setChecked(true);
        } else {
            r42.setChecked(false);
        }
        this.y.setOnCheckedChangeListener(new g());
    }

    @Override // android.app.Activity
    public boolean onOptionsItemSelected(MenuItem menuItem) {
        if (menuItem.getItemId() != 16908332) {
            return super.onOptionsItemSelected(menuItem);
        }
        finish();
        return true;
    }
}