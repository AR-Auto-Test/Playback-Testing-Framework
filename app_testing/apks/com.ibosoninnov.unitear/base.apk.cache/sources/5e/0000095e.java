package c.d.c.h.e.g;

import com.google.android.datatransport.Transformer;
import com.google.firebase.crashlytics.internal.model.CrashlyticsReport;
import com.google.firebase.crashlytics.internal.send.DataTransportCrashlyticsReportSender;
import java.nio.charset.Charset;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class a implements Transformer {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ a f4434a = new a();

    @Override // com.google.android.datatransport.Transformer
    public final Object apply(Object obj) {
        byte[] bytes;
        bytes = DataTransportCrashlyticsReportSender.TRANSFORM.reportToJson((CrashlyticsReport) obj).getBytes(Charset.forName("UTF-8"));
        return bytes;
    }
}