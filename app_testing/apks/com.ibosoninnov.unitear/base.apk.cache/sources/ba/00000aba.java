package c.e.b;

import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import com.ibosoninnov.unitear.ImageTrackingActivity;
import com.ibosoninnov.unitear.LoginWebviewActivity;
import com.ibosoninnov.unitear.R;
import com.ibosoninnov.unitear.SettingsActivity;
import com.ibosoninnov.unitear.activities.GuidanceActivity;
import java.util.Objects;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class o2 implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ImageTrackingActivity f5079b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ int f5080c;

    public /* synthetic */ o2(ImageTrackingActivity imageTrackingActivity, int i) {
        this.f5079b = imageTrackingActivity;
        this.f5080c = i;
    }

    @Override // java.lang.Runnable
    public final void run() {
        ImageTrackingActivity imageTrackingActivity = this.f5079b;
        int i = this.f5080c;
        Objects.requireNonNull(imageTrackingActivity);
        if (i == R.id.nav_visitunitear) {
            Intent intent = new Intent(imageTrackingActivity.D, LoginWebviewActivity.class);
            intent.setFlags(67108864);
            imageTrackingActivity.startActivity(intent);
        } else if (i == R.id.nav_rateapp) {
            String packageName = imageTrackingActivity.getPackageName();
            try {
                imageTrackingActivity.startActivity(new Intent("android.intent.action.VIEW", Uri.parse("market://details?id=" + packageName)));
            } catch (ActivityNotFoundException unused) {
                imageTrackingActivity.startActivity(new Intent("android.intent.action.VIEW", Uri.parse("https://play.google.com/store/apps/details?id=" + packageName)));
            }
        } else if (i == R.id.nav_guidance) {
            imageTrackingActivity.startActivity(new Intent(imageTrackingActivity, GuidanceActivity.class));
        } else if (i != R.id.nav_share) {
            if (i == R.id.nav_settings) {
                Context context = imageTrackingActivity.D;
                context.startActivity(new Intent(context, SettingsActivity.class));
            }
        } else {
            Intent intent2 = new Intent();
            intent2.setAction("android.intent.action.SEND");
            StringBuilder x = c.b.a.a.a.x("UniteAR - Create augmented reality without coding : https://play.google.com/store/apps/details?id=");
            x.append(imageTrackingActivity.getPackageName());
            intent2.putExtra("android.intent.extra.TEXT", x.toString());
            intent2.setType("text/plain");
            imageTrackingActivity.startActivity(intent2);
        }
    }
}