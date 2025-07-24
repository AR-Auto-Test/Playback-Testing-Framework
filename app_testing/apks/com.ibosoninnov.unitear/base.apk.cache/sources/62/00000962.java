package c.d.c.i.a;

import com.google.firebase.encoders.ValueEncoder;
import com.google.firebase.encoders.ValueEncoderContext;
import com.google.firebase.encoders.json.JsonDataEncoderBuilder;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class b implements ValueEncoder {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ b f4441a = new b();

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
    @Override // com.google.firebase.encoders.Encoder
    public final void encode(Object obj, ValueEncoderContext valueEncoderContext) {
        int i = JsonDataEncoderBuilder.f5645a;
        valueEncoderContext.add((String) obj);
    }
}