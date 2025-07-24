package c.e.b;

import android.util.Log;
import android.view.View;
import com.ibosoninnov.unitear.SplashActivity;

/* compiled from: SplashActivity.java */
/* loaded from: classes2.dex */
public class xe implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ SplashActivity f5419b;

    public xe(SplashActivity splashActivity) {
        this.f5419b = splashActivity;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        SplashActivity splashActivity = this.f5419b;
        int i = splashActivity.C;
        int i2 = splashActivity.w - 1;
        if (i < i2) {
            int i3 = i + 1;
            splashActivity.C = i3;
            splashActivity.t.setCurrentItem(i3);
        } else if (i == i2) {
            if (!splashActivity.E) {
                SplashActivity.v(splashActivity);
            } else {
                splashActivity.w();
            }
        }
        StringBuilder x = c.b.a.a.a.x("Next Pressed ");
        x.append(this.f5419b.C);
        x.append("/");
        x.append(this.f5419b.w);
        Log.d("SplashActivity", x.toString());
    }
}