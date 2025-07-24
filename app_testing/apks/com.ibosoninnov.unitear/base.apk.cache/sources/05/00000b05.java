package c.e.b;

import android.util.Log;
import c.e.b.cc;
import c.e.b.yd;
import com.google.ar.core.InstallActivity;
import com.google.firebase.crashlytics.internal.settings.SettingsJsonConstants;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: LoaderARContentSceneformARCore.java */
/* loaded from: classes2.dex */
public class sd implements cc.a {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ yd.e f5231a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ yd f5232b;

    public sd(yd ydVar, yd.e eVar) {
        this.f5232b = ydVar;
        this.f5231a = eVar;
    }

    @Override // c.e.b.cc.a
    public void a(String str) {
        try {
            this.f5231a.a(new JSONObject(str).getString(InstallActivity.MESSAGE_TYPE_KEY));
            this.f5232b.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.w7
                @Override // java.lang.Runnable
                public final void run() {
                    sd.this.f5232b.j();
                }
            });
        } catch (JSONException e2) {
            e2.printStackTrace();
        }
    }

    @Override // c.e.b.cc.a
    public void b(String str) {
        Log.d("LoaderARContentSceneformARCore", str);
        this.f5232b.B = str;
        try {
            JSONObject jSONObject = new JSONObject(str);
            if (jSONObject.getBoolean(SettingsJsonConstants.APP_STATUS_KEY)) {
                JSONArray jSONArray = jSONObject.getJSONObject("data").getJSONArray("arContent");
                this.f5232b.s = jSONArray.length();
                yd ydVar = this.f5232b;
                if (ydVar.s == 0) {
                    ydVar.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.t7
                        @Override // java.lang.Runnable
                        public final void run() {
                            sd.this.f5232b.j();
                        }
                    });
                }
                int length = jSONArray.length();
                for (int i = 0; i < length; i++) {
                    this.f5232b.g(jSONArray.getJSONObject(i));
                }
                return;
            }
            this.f5231a.a(jSONObject.getString(InstallActivity.MESSAGE_TYPE_KEY));
            this.f5232b.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.u7
                @Override // java.lang.Runnable
                public final void run() {
                    sd.this.f5232b.j();
                }
            });
        } catch (JSONException e2) {
            e2.printStackTrace();
            this.f5231a.a(e2.getMessage());
            this.f5232b.f5451b.runOnUiThread(new Runnable() { // from class: c.e.b.v7
                @Override // java.lang.Runnable
                public final void run() {
                    sd.this.f5232b.j();
                }
            });
        }
    }
}