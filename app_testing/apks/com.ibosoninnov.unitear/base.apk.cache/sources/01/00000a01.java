package c.e.b;

import android.view.View;
import com.ibosoninnov.unitear.ImageTrackingActivity;

/* compiled from: ImageTrackingActivity.java */
/* loaded from: classes2.dex */
public class fc implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ImageTrackingActivity f4753b;

    public fc(ImageTrackingActivity imageTrackingActivity) {
        this.f4753b = imageTrackingActivity;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        ImageTrackingActivity imageTrackingActivity = this.f4753b;
        imageTrackingActivity.Y0 = "";
        c.e.b.p000if.d dVar = imageTrackingActivity.C;
        dVar.f4872b.putString("custom_user", "");
        dVar.f4872b.apply();
        this.f4753b.G0.setVisibility(8);
        this.f4753b.J();
        this.f4753b.N0.c(false);
    }
}