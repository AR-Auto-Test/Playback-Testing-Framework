package c.e.b;

import android.util.Log;
import c.e.b.cc;
import c.e.b.hd;
import com.google.ar.core.InstallActivity;
import com.google.firebase.crashlytics.internal.settings.SettingsJsonConstants;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: LoaderARContentSceneform.java */
/* loaded from: classes2.dex */
public class kd implements cc.a {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ hd.g f4979a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ hd f4980b;

    public kd(hd hdVar, hd.g gVar) {
        this.f4980b = hdVar;
        this.f4979a = gVar;
    }

    @Override // c.e.b.cc.a
    public void a(String str) {
        try {
            this.f4980b.D = true;
            this.f4979a.a(new JSONObject(str).getString(InstallActivity.MESSAGE_TYPE_KEY));
            this.f4980b.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.o4
                @Override // java.lang.Runnable
                public final void run() {
                    kd.this.f4980b.k();
                }
            });
        } catch (JSONException e2) {
            e2.printStackTrace();
        }
    }

    @Override // c.e.b.cc.a
    public void b(String str) {
        Log.d("LoaderARContent", str);
        this.f4980b.B = str;
        try {
            JSONObject jSONObject = new JSONObject(str);
            if (jSONObject.getBoolean(SettingsJsonConstants.APP_STATUS_KEY)) {
                JSONArray jSONArray = jSONObject.getJSONObject("data").getJSONArray("arContent");
                this.f4980b.r = jSONArray.length();
                hd hdVar = this.f4980b;
                int i = hdVar.r;
                hdVar.s = i;
                hdVar.D = true;
                if (i == 0) {
                    hdVar.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.m4
                        @Override // java.lang.Runnable
                        public final void run() {
                            kd.this.f4980b.k();
                        }
                    });
                }
                int length = jSONArray.length();
                for (int i2 = 0; i2 < length; i2++) {
                    hd.a(this.f4980b, jSONArray.getJSONObject(i2), i2);
                }
                return;
            }
            this.f4980b.D = true;
            this.f4979a.a(jSONObject.getString(InstallActivity.MESSAGE_TYPE_KEY));
            this.f4980b.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.n4
                @Override // java.lang.Runnable
                public final void run() {
                    kd.this.f4980b.k();
                }
            });
        } catch (JSONException e2) {
            e2.printStackTrace();
            this.f4979a.a(e2.getMessage());
            this.f4980b.f4816g.runOnUiThread(new Runnable() { // from class: c.e.b.l4
                @Override // java.lang.Runnable
                public final void run() {
                    kd.this.f4980b.k();
                }
            });
        }
    }
}