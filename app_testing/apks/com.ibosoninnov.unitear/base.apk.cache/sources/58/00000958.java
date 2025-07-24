package c.d.c.h.e.e.a;

import android.util.JsonReader;
import com.google.firebase.crashlytics.internal.model.CrashlyticsReport;
import com.google.firebase.crashlytics.internal.model.serialization.CrashlyticsReportJsonTransform;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class e implements CrashlyticsReportJsonTransform.ObjectParser {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ e f4428a = new e();

    @Override // com.google.firebase.crashlytics.internal.model.serialization.CrashlyticsReportJsonTransform.ObjectParser
    public final Object parse(JsonReader jsonReader) {
        CrashlyticsReport.Session.Event.Application.Execution.BinaryImage parseEventBinaryImage;
        parseEventBinaryImage = CrashlyticsReportJsonTransform.parseEventBinaryImage(jsonReader);
        return parseEventBinaryImage;
    }
}