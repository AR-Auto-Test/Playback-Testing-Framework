package c.e.b;

import android.util.Log;
import c.e.b.cc;
import c.e.b.hd;
import com.google.ar.core.InstallActivity;
import com.google.firebase.crashlytics.internal.settings.SettingsJsonConstants;
import com.ibosoninnov.unitear.ImageTrackingActivity;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: LoaderARContentSceneform.java */
/* loaded from: classes2.dex */
public class jd implements cc.a {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ hd.g f4951a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ hd.f f4952b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ hd f4953c;

    public jd(hd hdVar, hd.g gVar, hd.f fVar) {
        this.f4953c = hdVar;
        this.f4951a = gVar;
        this.f4952b = fVar;
    }

    @Override // c.e.b.cc.a
    public void a(String str) {
    }

    @Override // c.e.b.cc.a
    public void b(String str) {
        Log.d("LoaderARContent", str);
        this.f4953c.B = str;
        try {
            JSONObject jSONObject = new JSONObject(str);
            if (jSONObject.getBoolean(SettingsJsonConstants.APP_STATUS_KEY)) {
                JSONArray jSONArray = jSONObject.getJSONObject("data").getJSONArray("arContent");
                this.f4953c.r = jSONArray.length();
                hd hdVar = this.f4953c;
                int i = hdVar.r;
                hdVar.s = i;
                hdVar.D = true;
                if (i == 0) {
                    hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.g4
                        @Override // java.lang.Runnable
                        public final void run() {
                            jd.this.f4953c.k();
                        }
                    });
                }
                int length = jSONArray.length();
                for (int i2 = 0; i2 < length; i2++) {
                    hd.a(this.f4953c, jSONArray.getJSONObject(i2), i2);
                }
                this.f4951a.a("");
                String string = jSONObject.getJSONObject("data").getJSONObject("target").getString("compressed_image_url");
                ImageTrackingActivity imageTrackingActivity = ((c1) this.f4952b).f4586a;
                int i3 = ImageTrackingActivity.r;
                imageTrackingActivity.F(string);
                return;
            }
            this.f4953c.D = true;
            this.f4951a.a(jSONObject.getString(InstallActivity.MESSAGE_TYPE_KEY));
            this.f4953c.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.i4
                @Override // java.lang.Runnable
                public final void run() {
                    jd.this.f4953c.k();
                }
            });
        } catch (JSONException e2) {
            e2.printStackTrace();
            this.f4951a.a(e2.getMessage());
            this.f4953c.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.h4
                @Override // java.lang.Runnable
                public final void run() {
                    jd.this.f4953c.k();
                }
            });
        }
    }
}