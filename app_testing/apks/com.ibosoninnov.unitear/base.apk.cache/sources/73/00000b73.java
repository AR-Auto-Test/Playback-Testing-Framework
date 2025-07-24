package c.e.b;

import android.os.Handler;
import android.view.View;
import android.view.animation.AlphaAnimation;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import androidx.viewpager.widget.ViewPager;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;
import com.ibosoninnov.unitear.SplashActivity;
import java.util.Objects;

/* compiled from: SplashActivity.java */
/* loaded from: classes2.dex */
public class ye implements ViewPager.i {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ SplashActivity f5475a;

    /* compiled from: SplashActivity.java */
    /* loaded from: classes2.dex */
    public class a implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ int f5476b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ ImageView f5477c;

        /* renamed from: d  reason: collision with root package name */
        public final /* synthetic */ Animation f5478d;

        /* renamed from: e  reason: collision with root package name */
        public final /* synthetic */ ImageView f5479e;

        /* renamed from: f  reason: collision with root package name */
        public final /* synthetic */ ImageView f5480f;

        /* compiled from: SplashActivity.java */
        /* renamed from: c.e.b.ye$a$a  reason: collision with other inner class name */
        /* loaded from: classes2.dex */
        public class RunnableC0090a implements Runnable {
            public RunnableC0090a() {
            }

            @Override // java.lang.Runnable
            public void run() {
                a.this.f5479e.startAnimation(AnimationUtils.loadAnimation(ye.this.f5475a, R.anim.translate_right));
            }
        }

        /* compiled from: SplashActivity.java */
        /* loaded from: classes2.dex */
        public class b implements Runnable {
            public b() {
            }

            @Override // java.lang.Runnable
            public void run() {
                a.this.f5479e.clearAnimation();
                a.this.f5479e.setImageResource(R.drawable.ar_floor);
            }
        }

        /* compiled from: SplashActivity.java */
        /* loaded from: classes2.dex */
        public class c implements Runnable {
            public c() {
            }

            @Override // java.lang.Runnable
            public void run() {
                a.this.f5479e.startAnimation(AnimationUtils.loadAnimation(ye.this.f5475a, R.anim.scaledown));
                a.this.f5480f.setVisibility(8);
                a.this.f5480f.clearAnimation();
            }
        }

        /* compiled from: SplashActivity.java */
        /* loaded from: classes2.dex */
        public class d implements Runnable {
            public d() {
            }

            @Override // java.lang.Runnable
            public void run() {
                a.this.f5479e.clearAnimation();
                a.this.f5479e.setImageResource(R.drawable.ar_floor);
                a aVar = a.this;
                aVar.f5477c.startAnimation(aVar.f5478d);
                a.this.f5477c.setVisibility(0);
            }
        }

        public a(int i, ImageView imageView, Animation animation, ImageView imageView2, ImageView imageView3) {
            this.f5476b = i;
            this.f5477c = imageView;
            this.f5478d = animation;
            this.f5479e = imageView2;
            this.f5480f = imageView3;
        }

        @Override // java.lang.Runnable
        public void run() {
            int i = this.f5476b;
            if (i == 1 || i == 2) {
                this.f5477c.startAnimation(this.f5478d);
                this.f5477c.setVisibility(0);
            }
            if (this.f5476b == 2) {
                new Handler().postDelayed(new RunnableC0090a(), 200L);
                new Handler().postDelayed(new b(), 800L);
                SplashActivity splashActivity = ye.this.f5475a;
                ImageView imageView = this.f5477c;
                int i2 = SplashActivity.r;
                Objects.requireNonNull(splashActivity);
                new Handler().postDelayed(new ze(splashActivity, imageView), 800L);
            }
            if (this.f5476b == 3) {
                AlphaAnimation alphaAnimation = new AlphaAnimation((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
                alphaAnimation.setDuration(300L);
                this.f5480f.startAnimation(alphaAnimation);
                this.f5480f.setVisibility(0);
                new Handler().postDelayed(new c(), 300L);
                new Handler().postDelayed(new d(), 600L);
            }
        }
    }

    public ye(SplashActivity splashActivity) {
        this.f5475a = splashActivity;
    }

    @Override // androidx.viewpager.widget.ViewPager.i
    public void onPageScrollStateChanged(int i) {
    }

    @Override // androidx.viewpager.widget.ViewPager.i
    public void onPageScrolled(int i, float f2, int i2) {
    }

    @Override // androidx.viewpager.widget.ViewPager.i
    public void onPageSelected(int i) {
        SplashActivity splashActivity;
        int i2 = 0;
        while (true) {
            splashActivity = this.f5475a;
            if (i2 >= splashActivity.w) {
                break;
            }
            ImageView imageView = splashActivity.x[i2];
            Object obj = b.j.c.a.f2074a;
            imageView.setImageDrawable(splashActivity.getDrawable(R.drawable.viewpager_dot_disabled));
            i2++;
        }
        ImageView imageView2 = splashActivity.x[i];
        Object obj2 = b.j.c.a.f2074a;
        imageView2.setImageDrawable(splashActivity.getDrawable(R.drawable.viewpager_dot_selected));
        c.e.b.hf.f fVar = this.f5475a.s.get(i);
        this.f5475a.y.setText(fVar.title);
        this.f5475a.z.setText(fVar.description);
        SplashActivity splashActivity2 = this.f5475a;
        if (i == splashActivity2.w - 1) {
            splashActivity2.B.setText(splashActivity2.getResources().getString(R.string.start_augmenting));
        } else {
            splashActivity2.B.setText(splashActivity2.getResources().getString(R.string.next));
        }
        if (i != 0) {
            View findViewWithTag = this.f5475a.t.findViewWithTag("tutorialview" + i);
            ImageView imageView3 = (ImageView) findViewWithTag.findViewById(R.id.iv_onboard);
            ImageView imageView4 = (ImageView) findViewWithTag.findViewById(R.id.iv_touch);
            ImageView imageView5 = (ImageView) findViewWithTag.findViewById(R.id.iv_phoneframe);
            Animation loadAnimation = AnimationUtils.loadAnimation(this.f5475a, R.anim.translate_up_phoneframe);
            imageView5.startAnimation(loadAnimation);
            imageView5.setVisibility(0);
            ImageView imageView6 = (ImageView) findViewWithTag.findViewById(R.id.iv_content);
            imageView6.setVisibility(4);
            Animation loadAnimation2 = AnimationUtils.loadAnimation(this.f5475a, R.anim.scaleup);
            loadAnimation2.setInterpolator(new c.e.b.p000if.i(0.30000001192092896d, 8.0d));
            imageView6.clearAnimation();
            if (i == 1) {
                imageView6.setImageResource(R.drawable.ar_objimage);
            }
            if (i == 2) {
                imageView6.setImageResource(R.drawable.ar_obj_ground);
                imageView3.setImageResource(R.drawable.ss_3);
            }
            if (i == 3) {
                imageView6.setImageResource(R.drawable.ar_objimage1);
                imageView3.setImageResource(R.drawable.argallery);
                imageView3.startAnimation(loadAnimation);
            }
            this.f5475a.F.removeCallbacksAndMessages(null);
            this.f5475a.F.postDelayed(new a(i, imageView6, loadAnimation2, imageView3, imageView4), 800L);
        }
        this.f5475a.C = i;
    }
}