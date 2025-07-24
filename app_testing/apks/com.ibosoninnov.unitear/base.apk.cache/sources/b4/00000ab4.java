package c.e.b;

import android.util.Log;
import c.e.b.ec;
import c.e.b.jc;
import com.google.ar.core.InstallActivity;
import com.google.firebase.crashlytics.internal.settings.SettingsJsonConstants;
import java.util.Objects;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/* compiled from: LoaderARContentGroundPlaneSceneform.java */
/* loaded from: classes2.dex */
public class nc implements ec.a {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ jc.c f5070a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ jc f5071b;

    public nc(jc jcVar, jc.c cVar) {
        this.f5071b = jcVar;
        this.f5070a = cVar;
    }

    @Override // c.e.b.ec.a
    public void a(String str) {
    }

    @Override // c.e.b.ec.a
    public void b(String str) {
        Objects.requireNonNull(this.f5071b);
        Log.d("LoaderARContentGroundPlaneSceneform", str);
        try {
            JSONObject jSONObject = new JSONObject(str);
            if (jSONObject.getBoolean(SettingsJsonConstants.APP_STATUS_KEY)) {
                JSONArray jSONArray = jSONObject.getJSONObject("data").getJSONObject("arBundle").getJSONArray("arContent");
                for (int i = 0; i < jSONArray.length(); i++) {
                    jc.b(this.f5071b, jSONArray.getJSONObject(i), i);
                }
                ((cb) this.f5070a).a("");
                return;
            }
            ((cb) this.f5070a).a(jSONObject.getString(InstallActivity.MESSAGE_TYPE_KEY));
        } catch (JSONException e2) {
            e2.printStackTrace();
            ((cb) this.f5070a).a(e2.getMessage());
        }
    }
}