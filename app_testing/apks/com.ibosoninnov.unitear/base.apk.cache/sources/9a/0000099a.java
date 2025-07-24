package c.e.b;

import android.app.Dialog;
import android.view.View;
import com.ibosoninnov.unitear.SplashActivity;

/* compiled from: SplashActivity.java */
/* loaded from: classes2.dex */
public class af implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Dialog f4555b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ SplashActivity f4556c;

    public af(SplashActivity splashActivity, Dialog dialog) {
        this.f4556c = splashActivity;
        this.f4555b = dialog;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        this.f4555b.dismiss();
        SplashActivity splashActivity = this.f4556c;
        int i = SplashActivity.r;
        splashActivity.w();
    }
}