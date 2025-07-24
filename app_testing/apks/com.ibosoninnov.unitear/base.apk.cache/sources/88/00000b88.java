package c.e.b;

import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import com.ibosoninnov.unitear.R;
import com.ibosoninnov.unitear.SplashActivity;

/* compiled from: SplashActivity.java */
/* loaded from: classes2.dex */
public class ze implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ImageView f5516b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ SplashActivity f5517c;

    public ze(SplashActivity splashActivity, ImageView imageView) {
        this.f5517c = splashActivity;
        this.f5516b = imageView;
    }

    @Override // java.lang.Runnable
    public void run() {
        this.f5516b.startAnimation(AnimationUtils.loadAnimation(this.f5517c, R.anim.translate_down_arobj));
    }
}