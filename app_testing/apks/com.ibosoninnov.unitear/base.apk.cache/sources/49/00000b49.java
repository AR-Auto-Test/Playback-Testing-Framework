package c.e.b;

import android.view.View;
import com.ibosoninnov.unitear.SplashActivity;

/* compiled from: SplashActivity.java */
/* loaded from: classes2.dex */
public class we implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ SplashActivity f5385b;

    public we(SplashActivity splashActivity) {
        this.f5385b = splashActivity;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        SplashActivity splashActivity = this.f5385b;
        if (!splashActivity.E) {
            SplashActivity.v(splashActivity);
        } else {
            splashActivity.w();
        }
    }
}