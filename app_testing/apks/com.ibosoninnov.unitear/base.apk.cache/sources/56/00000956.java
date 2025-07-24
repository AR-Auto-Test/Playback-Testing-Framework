package c.d.c.h.e.e.a;

import android.util.JsonReader;
import com.google.firebase.crashlytics.internal.model.CrashlyticsReport;
import com.google.firebase.crashlytics.internal.model.serialization.CrashlyticsReportJsonTransform;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class c implements CrashlyticsReportJsonTransform.ObjectParser {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ c f4426a = new c();

    @Override // com.google.firebase.crashlytics.internal.model.serialization.CrashlyticsReportJsonTransform.ObjectParser
    public final Object parse(JsonReader jsonReader) {
        CrashlyticsReport.Session.Event.Application.Execution.Thread parseEventThread;
        parseEventThread = CrashlyticsReportJsonTransform.parseEventThread(jsonReader);
        return parseEventThread;
    }
}