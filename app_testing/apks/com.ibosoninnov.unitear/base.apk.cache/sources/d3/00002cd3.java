package com.ibosoninnov.unitear;

import android.app.Dialog;
import android.content.Intent;
import android.graphics.drawable.ColorDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import androidx.viewpager.widget.ViewPager;
import b.b.c.h;
import b.j.c.a;
import c.e.b.af;
import c.e.b.ef.g;
import c.e.b.hf.f;
import c.e.b.p000if.d;
import c.e.b.we;
import c.e.b.xe;
import c.e.b.ye;
import java.util.ArrayList;

/* loaded from: classes2.dex */
public class SplashActivity extends h {
    public static final /* synthetic */ int r = 0;
    public TextView A;
    public Button B;
    public d D;
    public Handler F;
    public ArrayList<f> s;
    public ViewPager t;
    public LinearLayout u;
    public g v;
    public int w;
    public ImageView[] x;
    public TextView y;
    public TextView z;
    public int C = 0;
    public boolean E = false;

    public static void v(SplashActivity splashActivity) {
        if (splashActivity.E) {
            return;
        }
        Dialog dialog = new Dialog(splashActivity);
        dialog.setCancelable(false);
        dialog.getWindow().setBackgroundDrawable(new ColorDrawable(0));
        dialog.setContentView(R.layout.ar_warning);
        ((Button) dialog.findViewById(R.id.doneBtn)).setOnClickListener(new af(splashActivity, dialog));
        dialog.show();
        splashActivity.E = true;
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_splash);
        this.D = new d(this);
        this.F = new Handler();
        if (!this.D.f4871a.getBoolean("isStart", false)) {
            d dVar = this.D;
            dVar.f4872b.putLong("installedDate", System.currentTimeMillis());
            dVar.f4872b.apply();
            this.s = new ArrayList<>();
            int[] iArr = {R.string.splash_heading1, R.string.splash_heading2, R.string.splash_heading3, R.string.splash_heading4};
            int[] iArr2 = {R.string.splash_des1, R.string.splash_des2, R.string.splash_des3, R.string.splash_des4};
            int[] iArr3 = {R.drawable.ss_1, R.drawable.ss_2, R.drawable.ss_3, -1};
            for (int i = 0; i < 4; i++) {
                f fVar = new f();
                fVar.imageID = iArr3[i];
                fVar.title = getResources().getString(iArr[i]);
                fVar.description = getResources().getString(iArr2[i]);
                this.s.add(fVar);
            }
            this.w = this.s.size();
            this.t = (ViewPager) findViewById(R.id.pager_introduction);
            this.v = new g(this, this.s);
            this.u = (LinearLayout) findViewById(R.id.viewPagerCountDots);
            this.y = (TextView) findViewById(R.id.tv_header);
            this.z = (TextView) findViewById(R.id.tv_desc);
            TextView textView = (TextView) findViewById(R.id.tv_skipBtn);
            this.A = textView;
            textView.setOnClickListener(new we(this));
            Button button = (Button) findViewById(R.id.tv_button);
            this.B = button;
            button.setOnClickListener(new xe(this));
            this.t.setAdapter(this.v);
            ViewPager viewPager = this.t;
            ye yeVar = new ye(this);
            if (viewPager.W == null) {
                viewPager.W = new ArrayList();
            }
            viewPager.W.add(yeVar);
            int a2 = this.v.a();
            this.w = a2;
            this.x = new ImageView[a2];
            for (int i2 = 0; i2 < this.w; i2++) {
                this.x[i2] = new ImageView(this);
                ImageView imageView = this.x[i2];
                Object obj = a.f2074a;
                imageView.setImageDrawable(getDrawable(R.drawable.viewpager_dot_disabled));
                LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(-2, -2);
                layoutParams.setMargins(6, 0, 6, 0);
                this.u.addView(this.x[i2], layoutParams);
            }
            ImageView imageView2 = this.x[0];
            Object obj2 = a.f2074a;
            imageView2.setImageDrawable(getDrawable(R.drawable.viewpager_dot_selected));
        } else {
            w();
        }
        d dVar2 = this.D;
        dVar2.f4872b.putInt("launchCount", Integer.valueOf(this.D.f4871a.getInt("launchCount", 0)).intValue() + 1);
        dVar2.f4872b.apply();
    }

    @Override // b.q.b.d, android.app.Activity
    public void onRequestPermissionsResult(int i, String[] strArr, int[] iArr) {
        super.onRequestPermissionsResult(i, strArr, iArr);
        if (i != 11 || iArr.length <= 0) {
            return;
        }
        if (iArr[0] == 0) {
            Intent intent = new Intent(this, ImageTrackingActivity.class);
            intent.addFlags(536870912);
            startActivity(intent);
            finish();
            return;
        }
        Toast.makeText(this, getResources().getString(R.string.camera_permission_denied), 1).show();
    }

    public final void w() {
        d dVar = this.D;
        dVar.f4872b.putBoolean("isStart", true);
        dVar.f4872b.apply();
        if (Build.VERSION.SDK_INT > 28) {
            if (a.a(this, "android.permission.CAMERA") != 0) {
                b.j.b.a.c(this, new String[]{"android.permission.CAMERA", "android.permission.RECORD_AUDIO"}, 11);
                return;
            }
            Intent intent = new Intent(this, ImageTrackingActivity.class);
            intent.addFlags(536870912);
            startActivity(intent);
            finish();
        } else if (a.a(this, "android.permission.CAMERA") == 0 && a.a(this, "android.permission.WRITE_EXTERNAL_STORAGE") == 0) {
            Intent intent2 = new Intent(this, ImageTrackingActivity.class);
            intent2.addFlags(536870912);
            startActivity(intent2);
            finish();
        } else {
            b.j.b.a.c(this, new String[]{"android.permission.CAMERA", "android.permission.RECORD_AUDIO", "android.permission.WRITE_EXTERNAL_STORAGE"}, 11);
        }
    }
}