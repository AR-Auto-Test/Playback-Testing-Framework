package c.e.b;

import android.util.Log;
import android.widget.Toast;
import c.e.b.bf;
import c.e.b.hd;
import com.google.ar.core.InstallActivity;
import com.ibosoninnov.unitear.ImageTrackingActivity;
import java.util.Objects;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: ImageTrackingActivity.java */
/* loaded from: classes2.dex */
public class gc implements bf.a {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ImageTrackingActivity f4784a;

    public gc(ImageTrackingActivity imageTrackingActivity) {
        this.f4784a = imageTrackingActivity;
    }

    public void a(String str) {
        try {
            JSONObject jSONObject = new JSONObject(str);
            String string = jSONObject.getString("status_code");
            final String string2 = jSONObject.getString(InstallActivity.MESSAGE_TYPE_KEY);
            if (string.equals("UDT200") && this.f4784a.j0) {
                String string3 = jSONObject.getString("imageurl");
                final String string4 = jSONObject.getString("alphaId");
                Log.d("ImageTrackingActivity", "TargetFound " + string4 + " " + string3);
                ImageTrackingActivity imageTrackingActivity = this.f4784a;
                imageTrackingActivity.d0 = true;
                imageTrackingActivity.Z = false;
                imageTrackingActivity.F(string3);
                this.f4784a.runOnUiThread(new Runnable() { // from class: c.e.b.p0
                    @Override // java.lang.Runnable
                    public final void run() {
                        final gc gcVar = gc.this;
                        String str2 = string4;
                        ImageTrackingActivity imageTrackingActivity2 = gcVar.f4784a;
                        imageTrackingActivity2.A = new hd(str2, ac.f4547a.f4549c, imageTrackingActivity2.y, imageTrackingActivity2.v, imageTrackingActivity2, imageTrackingActivity2);
                        gcVar.f4784a.A.u(new hd.g() { // from class: c.e.b.q0
                            @Override // c.e.b.hd.g
                            public final void a(final String str3) {
                                final gc gcVar2 = gc.this;
                                Objects.requireNonNull(gcVar2);
                                if (str3.isEmpty()) {
                                    return;
                                }
                                gcVar2.f4784a.runOnUiThread(new Runnable() { // from class: c.e.b.o0
                                    @Override // java.lang.Runnable
                                    public final void run() {
                                        gc gcVar3 = gc.this;
                                        String str4 = str3;
                                        ImageTrackingActivity imageTrackingActivity3 = gcVar3.f4784a;
                                        int i = ImageTrackingActivity.r;
                                        imageTrackingActivity3.L(str4, 5000);
                                    }
                                });
                            }
                        });
                    }
                });
                this.f4784a.G();
                this.f4784a.O(string4);
                this.f4784a.N();
            } else if (string.equals("UDT202") || string.equals("UDT203") || string.equals("UDT270") || string.equals("UDT280")) {
                ImageTrackingActivity imageTrackingActivity2 = this.f4784a;
                int i = ImageTrackingActivity.r;
                imageTrackingActivity2.N();
                ImageTrackingActivity imageTrackingActivity3 = this.f4784a;
                imageTrackingActivity3.d0 = true;
                if (imageTrackingActivity3.C.f4871a.getBoolean("Audio", false)) {
                    this.f4784a.A0.start();
                }
                this.f4784a.runOnUiThread(new Runnable() { // from class: c.e.b.m0
                    @Override // java.lang.Runnable
                    public final void run() {
                        gc gcVar = gc.this;
                        String str2 = string2;
                        ImageTrackingActivity imageTrackingActivity4 = gcVar.f4784a;
                        int i2 = ImageTrackingActivity.r;
                        imageTrackingActivity4.L(str2, 5000);
                        gcVar.f4784a.A();
                    }
                });
            }
        } catch (JSONException e2) {
            e2.printStackTrace();
        }
        ImageTrackingActivity imageTrackingActivity4 = this.f4784a;
        if (imageTrackingActivity4.M == 1 && imageTrackingActivity4.d0) {
            imageTrackingActivity4.runOnUiThread(new Runnable() { // from class: c.e.b.r0
                @Override // java.lang.Runnable
                public final void run() {
                    ImageTrackingActivity imageTrackingActivity5 = gc.this.f4784a;
                    int i2 = ImageTrackingActivity.r;
                    imageTrackingActivity5.A();
                }
            });
        }
        this.f4784a.Z = false;
    }

    public void b(final String str) {
        ImageTrackingActivity imageTrackingActivity = this.f4784a;
        int i = ImageTrackingActivity.r;
        imageTrackingActivity.N();
        ImageTrackingActivity imageTrackingActivity2 = this.f4784a;
        imageTrackingActivity2.d0 = true;
        if (imageTrackingActivity2.C.f4871a.getBoolean("Audio", false)) {
            this.f4784a.A0.start();
        }
        if (str.contains("Failed to connect")) {
            return;
        }
        if (!str.toLowerCase().contains("thread interrupted")) {
            this.f4784a.runOnUiThread(new Runnable() { // from class: c.e.b.n0
                @Override // java.lang.Runnable
                public final void run() {
                    gc gcVar = gc.this;
                    String str2 = str;
                    Objects.requireNonNull(gcVar);
                    if (str2.toLowerCase().contains("timeout")) {
                        ImageTrackingActivity imageTrackingActivity3 = gcVar.f4784a;
                        int i2 = ImageTrackingActivity.r;
                        imageTrackingActivity3.L("Timeout", 5000);
                    } else if (str2.contains("502 Bad Gateway")) {
                        Toast.makeText(gcVar.f4784a.D, "Updating Server.. Try Later!", 0).show();
                    } else {
                        Toast.makeText(gcVar.f4784a.D, "Something went wrong.. Try Later!", 0).show();
                    }
                }
            });
        }
        ImageTrackingActivity imageTrackingActivity3 = this.f4784a;
        if (imageTrackingActivity3.M == 1 && imageTrackingActivity3.d0) {
            imageTrackingActivity3.runOnUiThread(new Runnable() { // from class: c.e.b.l0
                @Override // java.lang.Runnable
                public final void run() {
                    ImageTrackingActivity imageTrackingActivity4 = gc.this.f4784a;
                    int i2 = ImageTrackingActivity.r;
                    imageTrackingActivity4.A();
                }
            });
        }
        this.f4784a.Z = false;
    }
}