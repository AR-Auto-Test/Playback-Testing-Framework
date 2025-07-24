package com.google.protobuf;

import com.google.protobuf.Descriptors;
import com.google.protobuf.MessageLite;
import java.io.InputStream;

/* loaded from: classes2.dex */
public interface Message extends MessageLite, MessageOrBuilder {

    /* loaded from: classes2.dex */
    public interface Builder extends MessageLite.Builder, MessageOrBuilder {
        Builder addRepeatedField(Descriptors.FieldDescriptor fieldDescriptor, Object obj);

        @Override // 
        Message build();

        @Override // 
        Message buildPartial();

        @Override // 
        Builder clear();

        Builder clearField(Descriptors.FieldDescriptor fieldDescriptor);

        Builder clearOneof(Descriptors.OneofDescriptor oneofDescriptor);

        @Override // 
        Builder clone();

        Descriptors.Descriptor getDescriptorForType();

        Builder getFieldBuilder(Descriptors.FieldDescriptor fieldDescriptor);

        Builder getRepeatedFieldBuilder(Descriptors.FieldDescriptor fieldDescriptor, int i);

        @Override // 
        boolean mergeDelimitedFrom(InputStream inputStream);

        @Override // 
        boolean mergeDelimitedFrom(InputStream inputStream, ExtensionRegistryLite extensionRegistryLite);

        @Override // 
        Builder mergeFrom(ByteString byteString);

        @Override // 
        Builder mergeFrom(ByteString byteString, ExtensionRegistryLite extensionRegistryLite);

        @Override // 
        Builder mergeFrom(CodedInputStream codedInputStream);

        @Override // 
        Builder mergeFrom(CodedInputStream codedInputStream, ExtensionRegistryLite extensionRegistryLite);

        Builder mergeFrom(Message message);

        @Override // 
        Builder mergeFrom(InputStream inputStream);

        @Override // 
        Builder mergeFrom(InputStream inputStream, ExtensionRegistryLite extensionRegistryLite);

        @Override // 
        Builder mergeFrom(byte[] bArr);

        @Override // 
        Builder mergeFrom(byte[] bArr, int i, int i2);

        @Override // 
        Builder mergeFrom(byte[] bArr, int i, int i2, ExtensionRegistryLite extensionRegistryLite);

        @Override // 
        Builder mergeFrom(byte[] bArr, ExtensionRegistryLite extensionRegistryLite);

        Builder mergeUnknownFields(UnknownFieldSet unknownFieldSet);

        Builder newBuilderForField(Descriptors.FieldDescriptor fieldDescriptor);

        Builder setField(Descriptors.FieldDescriptor fieldDescriptor, Object obj);

        Builder setRepeatedField(Descriptors.FieldDescriptor fieldDescriptor, int i, Object obj);

        Builder setUnknownFields(UnknownFieldSet unknownFieldSet);
    }

    boolean equals(Object obj);

    @Override // 
    Parser<? extends Message> getParserForType();

    int hashCode();

    @Override // 
    Builder newBuilderForType();

    @Override // 
    Builder toBuilder();

    String toString();
}