package c.e.b;

import com.ibosoninnov.unitear.ImageTrackingActivity;
import java.util.Locale;
import java.util.TimerTask;

/* compiled from: ImageTrackingActivity.java */
/* loaded from: classes2.dex */
public class hc extends TimerTask {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ImageTrackingActivity f4809b;

    public hc(ImageTrackingActivity imageTrackingActivity) {
        this.f4809b = imageTrackingActivity;
    }

    @Override // java.util.TimerTask, java.lang.Runnable
    public void run() {
        this.f4809b.runOnUiThread(new Runnable() { // from class: c.e.b.t0
            @Override // java.lang.Runnable
            public final void run() {
                hc hcVar = hc.this;
                ImageTrackingActivity imageTrackingActivity = hcVar.f4809b;
                if (imageTrackingActivity.f0) {
                    return;
                }
                int i = imageTrackingActivity.N + 1;
                imageTrackingActivity.N = i;
                if (i >= 60) {
                    imageTrackingActivity.N = 0;
                    imageTrackingActivity.O++;
                }
                hcVar.f4809b.Q0.setText(String.format(Locale.ENGLISH, "%02d:%02d", Integer.valueOf(imageTrackingActivity.O), Integer.valueOf(hcVar.f4809b.N)));
            }
        });
    }
}